#include "diagnostics/timing.h"
#include "runtime.h"
// 根编码、设备搬运与 BF16 更新共用的原生加速器实现。
#include <ATen/ops/embedding_bag.h>
#include <ATen/record_function.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#ifdef HJ_CUDA
// 必须先定义 CUDA_VERSION，避免 CUDAGraph 与发行库的条件成员布局不同。
// clang-format off
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
// clang-format on
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace hj {
using namespace packing;
using Clock = std::chrono::steady_clock;
namespace {
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::vector<std::shared_ptr<Root>> roots_of(const std::vector<std::shared_ptr<Node>> &nodes) {
  std::vector<std::shared_ptr<Root>> roots;
  for (auto &n : nodes)
    if (std::find(roots.begin(), roots.end(), n->root) == roots.end())
      roots.push_back(n->root);
  return roots;
}
#ifdef HJ_CUDA
void cuda_check(cudaError_t e) {
  TORCH_CHECK(e == cudaSuccess, "CUDA inference: ", cudaGetErrorString(e));
}
void driver_check(CUresult e) {
  if (e != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(e, &message);
    TORCH_CHECK(false, "CUDA driver: ", message ? message : "unknown error");
  }
}
// 类型化投影与均值按真实长度计算；累加只在寄存器中，输出张量仍为 BF16。
class DeviceKernels {
  CUmodule module = nullptr;
  CUfunction scatter_, store_, project_, update_, targets_, binary_, reduce_, residual_norm_,
      attention_;
  T receipt;

public:
  DeviceKernels() {
#ifdef _WIN32
    auto name = std::wstring(L"nvrtc64_") + std::to_wstring(CUDA_VERSION / 1000) + L"0_0.dll";
    auto library = LoadLibraryW(name.c_str());
    auto symbol = [&](const char *s) {
      return reinterpret_cast<void *>(GetProcAddress(library, s));
    };
#else
    auto name = std::string("libnvrtc.so.") + std::to_string(CUDA_VERSION / 1000);
    auto library = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    auto symbol = [&](const char *s) { return dlsym(library, s); };
#endif
    TORCH_CHECK(library, "NVRTC library is required for native input graphs");
    using Program = void *;
    auto create =
        reinterpret_cast<int (*)(Program *, const char *, const char *, int, const char *const *,
                                 const char *const *)>(symbol("nvrtcCreateProgram"));
    auto compile =
        reinterpret_cast<int (*)(Program, int, const char *const *)>(symbol("nvrtcCompileProgram"));
    auto log_size = reinterpret_cast<int (*)(Program, size_t *)>(symbol("nvrtcGetProgramLogSize"));
    auto log = reinterpret_cast<int (*)(Program, char *)>(symbol("nvrtcGetProgramLog"));
    auto ptx_size = reinterpret_cast<int (*)(Program, size_t *)>(symbol("nvrtcGetPTXSize"));
    auto ptx = reinterpret_cast<int (*)(Program, char *)>(symbol("nvrtcGetPTX"));
    auto destroy = reinterpret_cast<int (*)(Program *)>(symbol("nvrtcDestroyProgram"));
    TORCH_CHECK(create && compile && log_size && log && ptx_size && ptx && destroy,
                "incomplete NVRTC C API");
    const char *source = R"cuda(
typedef unsigned long long U;
typedef unsigned short B;
struct Field { U src,count,dst,cap,pad,unit,begin,end; };
__device__ float value(B x) { return __uint_as_float((unsigned)x<<16); }
__device__ B rounded(float x) {
  unsigned u=__float_as_uint(x);
  if((u&0x7fffffff)>0x7f800000) return (B)((u>>16)|0x40);
  return (B)((u+0x7fff+((u>>16)&1))>>16);
}
__device__ float r(float x) { return value(rounded(x)); }
extern "C" __global__ void attention(const B *qkv,const int *cu,const int *used,B *out,
                                    int roots,int heads,int dim) {
  __shared__ float scores[512], sums[256];
  int q=blockIdx.x, h=blockIdx.y, t=threadIdx.x, group=t/4, lane=t%4;
  int root=0, end=roots;
  while(root+1<end) { int mid=(root+end)/2; if(q>=cu[mid]) root=mid; else end=mid; }
  int start=cu[root], length=used?used[root]:cu[root+1]-start;
  if(q>=start+length) { if(t<dim) out[(q*heads+h)*dim+t]=rounded(0); return; }
  for(int base=0;base<length;base+=64) {
    int key=base+group;
    float sum=0;
    if(key<length)
      for(int d=lane;d<dim;d+=4)
        sum+=value(qkv[(q*3*heads+h)*dim+d])*
             value(qkv[(((start+key)*3+1)*heads+h)*dim+d]);
    sum+=__shfl_down_sync(0xffffffff,sum,2,4);
    sum+=__shfl_down_sync(0xffffffff,sum,1,4);
    if(!lane && key<length) scores[key]=sum*rsqrtf((float)dim);
  }
  __syncthreads();
  float maximum=-1.0f/0.0f;
  for(int k=t;k<length;k+=256) maximum=fmaxf(maximum,scores[k]);
  sums[t]=maximum;
  __syncthreads();
  for(int n=128;n;n/=2) { if(t<n) sums[t]=fmaxf(sums[t],sums[t+n]); __syncthreads(); }
  maximum=sums[0];
  __syncthreads();
  float sum=0;
  for(int k=t;k<length;k+=256) { float e=expf(scores[k]-maximum); scores[k]=e; sum+=e; }
  sums[t]=sum;
  __syncthreads();
  for(int n=128;n;n/=2) { if(t<n) sums[t]+=sums[t+n]; __syncthreads(); }
  if(t<dim) {
    float v=0;
    for(int k=0;k<length;++k)
      v+=(scores[k]/sums[0])*value(qkv[(((start+k)*3+2)*heads+h)*dim+t]);
    out[(q*heads+h)*dim+t]=rounded(v);
  }
}
__device__ float row_sum(float sum,float *scratch,bool whole_block) {
  for(int n=16;n;n/=2) sum+=__shfl_down_sync(0xffffffff,sum,n);
  if(!whole_block) return __shfl_sync(0xffffffff,sum,0);
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  if(!lane) scratch[warp]=sum;
  __syncthreads();
  if(!warp) {
    sum=lane<8?scratch[lane]:0;
    for(int n=4;n;n/=2) sum+=__shfl_down_sync(0xffffffff,sum,n);
    if(!lane) scratch[0]=sum;
  }
  __syncthreads();
  float result=scratch[0];
  __syncthreads();
  return result;
}
// 大表每个 warp 独立处理一行；小表每行一块以保留 SM 并行度。残差和统计仍是 BF16。
extern "C" __global__ void residual_norm(const B *a,const B *b,const B *w,const B *bias,
                                        B *x,B *y,B *mean,B *inverse,int width,int rows,float eps) {
  __shared__ float scratch[8];
  bool whole_block=rows<=32;
  int lane=whole_block?threadIdx.x:threadIdx.x%32;
  int row=whole_block?blockIdx.x:blockIdx.x*4+threadIdx.x/32, base=row*width;
  int stride=whole_block?256:32;
  if(row>=rows) return;
  float sum=0;
  for(int j=lane;j<width;j+=stride) {
    B v=rounded(value(a[base+j])+value(b[base+j]));
    x[base+j]=v;
    sum+=value(v);
  }
  float m=row_sum(sum,scratch,whole_block)/width;
  sum=0;
  for(int j=lane;j<width;j+=stride) { float d=value(x[base+j])-m; sum+=d*d; }
  float inv=rsqrtf(row_sum(sum,scratch,whole_block)/width+eps);
  if(!lane) { mean[row]=rounded(m); inverse[row]=rounded(inv); }
  for(int j=lane;j<width;j+=stride)
    y[base+j]=rounded(((value(x[base+j])-m)*inv)*value(w[j])+value(bias[j]));
}
// 每个显式中间结果仍按 BF16 舍入；只融合搬运，不改变补偿更新的算式。
extern "C" __global__ void update(const B *g,B *p,B *m,B *v,B *mr,B *vr,B *pr,const B *norm,
                                  int *valid,const B *loss,U count,float lr,float bc1,float bc2,
                                  const U *control,int commit) {
  if(control) {
    lr=__uint_as_float((unsigned)control[1]);
    bc1=__uint_as_float((unsigned)control[2]);
    bc2=__uint_as_float((unsigned)control[3]);
  }
  U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count||(commit&&(!*valid||(control&&!control[0])))) return;
  if(i==0&&!commit) {
    ((B *)(valid+1))[0]=loss?*loss:0;
    if(loss&&!isfinite(value(*loss))) atomicExch(valid,0);
  }
  float n=value(*norm), grad=value(g[i]);
  float a=r(grad/fmaxf(n,1.0f)), old=value(p[i]);
  float dm=r(value(mr[i])+r(r(a-value(m[i]))*0.1f));
  float mn=r(value(m[i])+dm), mc=r(dm+r(value(m[i])-mn));
  float dv=r(value(vr[i])+r(r(r(a*a)-value(v[i]))*0.001f));
  float vn=r(value(v[i])+dv), vc=r(dv+r(value(v[i])-vn));
  float denominator=r(r(sqrtf(r(vn*bc2)))+1e-8f);
  float change=r(-lr*r(r(r(mn*bc1)/denominator)+r(0.01f*old)));
  float dp=r(value(pr[i])+change), pn=r(old+dp), pc=r(dp+r(old-pn));
  if(commit) {
    p[i]=rounded(pn); m[i]=rounded(mn); v[i]=rounded(vn);
    mr[i]=rounded(mc); vr[i]=rounded(vc); pr[i]=rounded(pc);
  } else if(!isfinite(n)||!isfinite(grad)||!isfinite(a)||!isfinite(pn)||!isfinite(mn)||
            !isfinite(vn)||!isfinite(mc)||!isfinite(vc)||!isfinite(pc)) atomicExch(valid,0);
}
// 原始整数位在设备解包成精确零一输入；连续槽独立处理，不参与身份转换。
extern "C" __global__ void binary(const U *raw,B *out,U count,int typed) {
  U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
  int width=typed?3944:4096;
  if(i>=count*(U)width) return;
  U row=i/width,col=i%width;
  U word; int bit;
  if(typed&&col>=3840) {
    word=raw[row*75+(col<3892?60:61)]; bit=(int)(col-(col<3892?3840:3892));
  } else {
    word=raw[row*(typed?75:64)+col/64]; word=(word<<1)^(0ULL-(word>>63));
    bit=(int)(col%64);
  }
  out[i]=((word>>bit)&1)?0x3f80:0;
}
// CSR 归约只读真实区间；同一个内核用于参数均值和 gather 的转置。
extern "C" __global__ void reduce(const B *data,const long long *ids,const long long *offsets,
                                  B *out,U rows,int width,int mean) {
  U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*(U)width) return;
  U row=i/width,col=i%width;
  long long begin=offsets[row],end=offsets[row+1];
  float sum=0;
  for(long long k=begin;k<end;++k) sum+=value(data[(U)ids[k]*width+col]);
  if(mean&&end>begin) sum/=(float)(end-begin);
  out[i]=rounded(sum);
}
// 同席位短轨迹按真实顺序反向递推；概率标量只在寄存器中使用 double。
extern "C" __global__ void targets(const B *mass,const B *values,const long long *records,
                                   const long long *blocks,B *out,long long count) {
  U bi=(U)blockIdx.x*blockDim.x+threadIdx.x;
  long long begin=blocks[bi*3],end=blocks[bi*3+1],bootstrap=blocks[bi*3+2];
  double correction=0,following=bootstrap<0?0:value(values[bootstrap]),following_target=following;
  for(long long i=end;i-->begin;) {
    const long long *s=records+i*6;
    double total=0;
    for(long long j=0;j<s[1];++j) total+=value(mass[s[0]+j]);
    double chosen=value(mass[s[0]+s[2]]);
    double logp=chosen>0?log(chosen/total):-__longlong_as_double(0x7ff0000000000000LL);
    double ratio=fmin(1.0,exp(logp-__longlong_as_double(s[5])));
    double v=value(values[i]),d=s[4]?0:1;
    correction=ratio*((double)s[3]+d*following-v)+d*ratio*correction;
    double target=v+correction;
    out[i]=rounded((float)target);
    out[count+i]=rounded((float)(ratio*((double)s[3]+d*following_target-v)));
    out[count*2+i]=rounded((float)logp);
    following=v; following_target=target;
  }
}
// 按真实行和置位数量求和，不扫描容量尾部的填充索引。
extern "C" __global__ void project(const Field *f,const U *data,const B *weight,B *out,B *continuous,
                                   int width,int capacity,int field,int typed) {
  U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
  U row=i/width,col=i%width;
  if(row>=(U)capacity) return;
  U count=f[field].count/(typed?75:64);
  float sum=0;
  if(row<count) {
    const U *p=data+row*(typed?75:64);
    for(int j=0;j<(typed?62:64);++j) {
      U bits=p[j];
      if(!typed||j<60) bits=(bits<<1)^(0ULL-(bits>>63));
      int base=j*64;
      if(typed && j>=60) base=j==60?3840:3892;
      while(bits) {
        int bit=__ffsll((long long)bits)-1;
        sum+=value(weight[(U)(base+bit)*width+col]);
        bits&=bits-1;
      }
    }
    if(typed) for(int c=col;c<52;c+=width) continuous[row*52+c]=((const B *)p)[248+c];
  } else if(typed) for(int c=col;c<52;c+=width) continuous[row*52+c]=0;
  out[i]=rounded(sum);
}
extern "C" __global__ void scatter(const unsigned char *src, unsigned char *dst, int fields) {
  const Field *f = (const Field *)src;
  int k = 0;
  while (k < fields && blockIdx.x >= f[k].end) ++k;
  if (k == fields) return;
  U i = (U)(blockIdx.x - f[k].begin) * blockDim.x + threadIdx.x;
  if (i >= f[k].cap) return;
  if (f[k].unit == 8)
    ((U *)(dst + f[k].dst))[i] = i < f[k].count ? ((const U *)(src + f[k].src))[i] : f[k].pad;
  else if (f[k].unit == 4)
    ((unsigned *)(dst + f[k].dst))[i] = i < f[k].count ? ((const unsigned *)(src + f[k].src))[i] : (unsigned)f[k].pad;
  else
    ((B *)(dst + f[k].dst))[i] = i < f[k].count ? ((const B *)(src + f[k].src))[i] : (B)f[k].pad;
}
// 描述符为输出实体指针、输出上下文指针、实体数、根的 global token 位置。
extern "C" __global__ void store(const U *r, const B *tokens, const B *context, int width, int capacity, int roots) {
  U i=(U)blockIdx.x*blockDim.x+threadIdx.x;
  U row=i/(width*2), col=i%(width*2);
  if (row<(U)capacity) {
    int lo=0, hi=roots;
    while(lo<hi) {
      int mid=(lo+hi)/2;
      if(r[mid*4+3]+1<=row) lo=mid+1; else hi=mid;
    }
    int k=lo-1;
    if(k>=0) {
      U n=r[k*4+2], off=r[k*4+3]+1;
      if(row<off+n) ((B *)r[k*4])[(row-off)*width*2+col]=tokens[i];
    }
  }
  if (i<(U)roots*width) {
    U k=i/width, c=i%width;
    if(r[k*4+1]) ((B *)r[k*4+1])[c]=context[i];
  }
}
)cuda";
    Program program = nullptr;
    TORCH_CHECK(create(&program, source, "haojie_kernels.cu", 0, nullptr, nullptr) == 0,
                "NVRTC program creation failed");
    int major = 0, minor = 0, current_device = 0;
    cuda_check(cudaGetDevice(&current_device));
    cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
    auto architecture =
        std::string("--gpu-architecture=compute_") + std::to_string(major) + std::to_string(minor);
    const char *options[] = {architecture.c_str(), "--std=c++17", "--fmad=false"};
    int status = compile(program, 3, options);
    size_t size = 0;
    if (status) {
      log_size(program, &size);
      std::string message(size, '\0');
      log(program, message.data());
      destroy(&program);
      TORCH_CHECK(false, "NVRTC: ", message);
    }
    TORCH_CHECK(ptx_size(program, &size) == 0, "NVRTC PTX size failed");
    std::string code(size, '\0');
    TORCH_CHECK(ptx(program, code.data()) == 0, "NVRTC PTX failed");
    destroy(&program);
    driver_check(cuModuleLoadData(&module, code.data()));
    driver_check(cuModuleGetFunction(&scatter_, module, "scatter"));
    driver_check(cuModuleGetFunction(&store_, module, "store"));
    driver_check(cuModuleGetFunction(&project_, module, "project"));
    driver_check(cuModuleGetFunction(&update_, module, "update"));
    driver_check(cuModuleGetFunction(&targets_, module, "targets"));
    driver_check(cuModuleGetFunction(&binary_, module, "binary"));
    driver_check(cuModuleGetFunction(&reduce_, module, "reduce"));
    driver_check(cuModuleGetFunction(&residual_norm_, module, "residual_norm"));
    driver_check(cuModuleGetFunction(&attention_, module, "attention"));
    receipt = torch::empty(
        {8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
  }
  ~DeviceKernels() {
    if (module)
      cuModuleUnload(module);
  }
  T attention(const T &qkv, const T &cu, const T &used) {
    int roots = int(cu.numel() - 1), heads = int(qkv.size(2)), dim = int(qkv.size(3));
    auto out = torch::empty({qkv.size(0), heads, dim}, qkv.options());
    auto q = qkv.data_ptr(), c = cu.data_ptr(), o = out.data_ptr();
    void *u = used.defined() ? used.data_ptr() : nullptr;
    void *args[] = {&q, &c, &u, &o, &roots, &heads, &dim};
    driver_check(cuLaunchKernel(attention_, unsigned(qkv.size(0)), heads, 1, 256, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
    return out;
  }
  std::vector<T> residual_norm(const T &a, const T &b, const T &weight, const T &bias,
                               double epsilon) {
    auto x = torch::empty_like(a), y = torch::empty_like(a);
    auto mean = torch::empty({a.size(0), 1}, a.options()), inverse = torch::empty_like(mean);
    auto ap = a.data_ptr(), bp = b.data_ptr(), wp = weight.data_ptr(), cp = bias.data_ptr(),
         xp = x.data_ptr(), yp = y.data_ptr(), mp = mean.data_ptr(), ip = inverse.data_ptr();
    int width = int(a.size(1)), rows = int(a.size(0));
    float eps = float(epsilon);
    void *args[] = {&ap, &bp, &wp, &cp, &xp, &yp, &mp, &ip, &width, &rows, &eps};
    driver_check(cuLaunchKernel(residual_norm_, unsigned(rows <= 32 ? rows : (rows + 3) / 4), 1, 1,
                                rows <= 32 ? 256 : 128, 1, 1, 0, c10::cuda::getCurrentCUDAStream(),
                                args, nullptr));
    return {x, y, mean, inverse};
  }
  T checked_update(const T &gradient, const T &parameters, const std::vector<T> &state,
                   const T &norm, double rate, uint64_t step, const T &loss, const T &control) {
    auto count = uint64_t(parameters.numel());
    auto valid = torch::ones({2}, parameters.options().dtype(torch::kInt32));
    auto g = gradient.data_ptr(), p = parameters.data_ptr(), m = state[0].data_ptr(),
         v = state[1].data_ptr(), mr = state[2].data_ptr(), vr = state[3].data_ptr(),
         pr = state[4].data_ptr(), n = norm.data_ptr(), ok = valid.data_ptr();
    void *loss_data = loss.defined() ? loss.data_ptr() : nullptr;
    // 与 ATen 的 CPU 标量除法相同：先在 double 中求倒数，再转为设备 opmath。
    float lr = float(rate), bc1 = float(1 / (1 - std::pow(0.9, double(step)))),
          bc2 = float(1 / (1 - std::pow(0.999, double(step))));
    void *control_data = control.defined() ? control.data_ptr() : nullptr;
    int commit = 0;
    void *args[] = {&g,  &p,         &m,     &v,  &mr,  &vr,  &pr,           &n,
                    &ok, &loss_data, &count, &lr, &bc1, &bc2, &control_data, &commit};
    // 第一遍仅检查；第二遍复算并提交，省去六组全参数提案的写入和重读。
    for (commit = 0; commit < 2; ++commit)
      driver_check(cuLaunchKernel(update_, unsigned((count + 255) / 256), 1, 1, 256, 1, 1, 0,
                                  c10::cuda::getCurrentCUDAStream(), args, nullptr));
    return valid;
  }
  UpdateResult update(const T &gradient, const T &parameters, const std::vector<T> &state,
                      const T &norm, double rate, uint64_t step, const T &loss) {
    auto valid = checked_update(gradient, parameters, state, norm, rate, step, loss, {});
    auto ok = valid.data_ptr();
    cuda_check(cudaMemcpyAsync(receipt.data_ptr(), ok, 8, cudaMemcpyDeviceToHost,
                               c10::cuda::getCurrentCUDAStream()));
    cuda_check(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));
    auto bytes = receipt.data_ptr<uint8_t>();
    int accepted;
    std::memcpy(&accepted, bytes, 4);
    if (!accepted)
      throw std::runtime_error("non-finite BF16 update; parameters and optimizer not committed");
    BF loss_value;
    std::memcpy(&loss_value.x, bytes + 4, 2);
    return {{}, double(float(loss_value)), true};
  }
  T targets(const T &mass, const T &values, const T &records, const T &blocks) {
    auto count = records.numel() / 6;
    auto output = torch::zeros({3, count}, values.options());
    auto p = mass.data_ptr(), v = values.data_ptr(), r = records.data_ptr(), b = blocks.data_ptr(),
         o = output.data_ptr();
    void *args[] = {&p, &v, &r, &b, &o, &count};
    driver_check(cuLaunchKernel(targets_, unsigned(blocks.numel() / 3), 1, 1, 1, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
    return output;
  }
  T captured_update(const T &gradient, const T &parameters, const std::vector<T> &state,
                    const T &norm, const T &loss, const T &control) {
    return checked_update(gradient, parameters, state, norm, 0, 1, loss, control);
  }
  T binary(const T &raw, bool typed) {
    auto count = uint64_t(raw.size(0));
    int t = typed, width = typed ? 3944 : 4096;
    auto output = torch::empty({int64_t(count), width}, raw.options().dtype(torch::kBFloat16));
    auto r = raw.data_ptr(), o = output.data_ptr();
    void *args[] = {&r, &o, &count, &t};
    driver_check(cuLaunchKernel(binary_, unsigned((count * width + 255) / 256), 1, 1, 256, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
    return output;
  }
  T reduce(const T &data, const T &ids, const T &offsets, bool mean) {
    auto count = uint64_t(offsets.numel() - 1);
    int width = int(data.size(1)), m = mean;
    auto output = torch::empty({int64_t(count), width}, data.options());
    auto d = data.data_ptr(), i = ids.data_ptr(), f = offsets.data_ptr(), o = output.data_ptr();
    void *args[] = {&d, &i, &f, &o, &count, &width, &m};
    driver_check(cuLaunchKernel(reduce_, unsigned((count * width + 255) / 256), 1, 1, 256, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
    return output;
  }
  void scatter(const T &src, const T &dst, int fields, int blocks) {
    auto s = src.data_ptr(), d = dst.data_ptr();
    void *args[] = {&s, &d, &fields};
    driver_check(cuLaunchKernel(scatter_, blocks, 1, 1, 256, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
  }
  void cache(const T &descriptors, const T &tokens, const T &context, int width, int capacity) {
    auto r = descriptors.data_ptr(), t = tokens.data_ptr(), c = context.data_ptr();
    int roots = int(descriptors.numel() / 4);
    void *args[] = {&r, &t, &c, &width, &capacity, &roots};
    auto blocks = (std::max<int64_t>(int64_t(capacity) * width * 2, roots * width) + 255) / 256;
    driver_check(cuLaunchKernel(store_, unsigned(blocks), 1, 1, 256, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
  }
  T project(const T &input, const T &data, const T &weights, const T &continuous, int width,
            int capacity, int field, bool typed) {
    auto out = torch::empty({capacity, width}, weights.options());
    auto f = input.data_ptr(), d = data.data_ptr(), w = weights.data_ptr(), o = out.data_ptr();
    void *c = typed ? continuous.data_ptr() : nullptr;
    int t = typed;
    void *args[] = {&f, &d, &w, &o, &c, &width, &capacity, &field, &t};
    driver_check(cuLaunchKernel(project_, (capacity * width + 255) / 256, 1, 1, 256, 1, 1, 0,
                                c10::cuda::getCurrentCUDAStream(), args, nullptr));
    return out;
  }
};
constexpr size_t Header = 32 * 64, ArenaBytes = 8 * 1024 * 1024;
struct Field {
  uint64_t src, count, dst, cap, pad, unit, begin, end;
};
struct Table {
  int64_t count;
  torch::ScalarType type;
  uint64_t padding;
  size_t offset;
  T tensor;
};
struct Upload {
  std::vector<Field> fields;
  size_t next = Header;
  int blocks = 0;
  uint8_t *host;
  explicit Upload(const T &pinned) : host(pinned.data_ptr<uint8_t>()) {}
  void put(const Table &t, const void *values, size_t count) {
    TORCH_CHECK(count <= size_t(t.count) && fields.size() < 32, "graph field overflow");
    auto unit = c10::elementSize(t.type);
    next = (next + 7) & ~size_t(7);
    TORCH_CHECK(next + count * unit <= ArenaBytes, "graph upload overflow");
    if (count)
      std::memcpy(host + next, values, count * unit);
    int end = blocks + int((t.count + 255) / 256);
    fields.push_back({next, count, t.offset, uint64_t(t.count), t.padding, unit, uint64_t(blocks),
                      uint64_t(end)});
    blocks = end;
    next += count * unit;
  }
  template <class V> void put(const Table &t, const std::vector<V> &v) {
    put(t, v.data(), v.size());
  }
  void done() {
    std::memcpy(host, fields.data(), fields.size() * sizeof(Field));
  }
};
struct Layout {
  size_t next = 0;
  const T &arena;
  explicit Layout(const T &a) : arena(a) {}
  Table add(int64_t count, torch::ScalarType type = torch::kInt64, uint64_t padding = 0) {
    next = (next + 7) & ~size_t(7);
    auto bytes = count * c10::elementSize(type);
    TORCH_CHECK(next + bytes <= ArenaBytes, "graph arena overflow");
    Table t{count, type, padding, next, arena.narrow(0, next, bytes).view(type)};
    next += bytes;
    return t;
  }
};
struct Bucket {
  int64_t nodes, candidates, rows, maximum;
  Table row_data, row_values, descriptors, global_ids, row_ids, kinds, gather, cumulative,
      key_lengths, global_positions;
  int fields = 0, copy_blocks = 0;
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  Bucket(int64_t n, int64_t c, const T &arena)
      : nodes(n), candidates(c), rows(n), maximum(std::min<int64_t>(n, 1024)) {
    Layout l(arena);
    row_data = l.add(n * 75);
    row_values = l.add(n * 52, torch::kBFloat16);
    descriptors = l.add(c * 4);
    global_ids = l.add(c);
    row_ids = l.add(n);
    kinds = l.add(n);
    gather = l.add(n);
    cumulative = l.add(c + 1, torch::kInt32);
    key_lengths = l.add(c, torch::kInt32, 1);
    global_positions = l.add(c);
  }
};
struct CacheWrite {
  std::shared_ptr<Root> root;
  int64_t offset, entities;
};
struct Slot {
  c10::cuda::CUDAStream stream;
  T pinned, input, arena;
  T output, host_output, graph_output;
  int64_t used = 0;
  std::map<std::pair<int64_t, int64_t>, std::unique_ptr<Bucket>> buckets;
  std::vector<CacheWrite> writes;
  std::shared_ptr<InferenceFlight> flight;
  Clock::time_point start;
  diagnostic::GpuSpan timing;
  double submitted_ms = 0;
  uint64_t token_count = 0;
  size_t bytes = 0, capacity_bytes = 0;
  bool active = false;
  std::atomic<uint64_t> *pending = nullptr;
  std::function<void()> *wake = nullptr;
  explicit Slot(const torch::Device &device)
      : stream(c10::cuda::getStreamFromPool(true, device.index() < 0 ? 0 : device.index())) {
    auto byte = torch::TensorOptions().dtype(torch::kUInt8);
    pinned = torch::empty({ArenaBytes}, byte.device(torch::kCPU).pinned_memory(true));
    input = torch::empty({ArenaBytes}, byte.device(device));
    arena = torch::empty({ArenaBytes}, byte.device(device));
  }
  void reserve(int64_t count, const torch::TensorOptions &options) {
    if (output.defined() && output.numel() >= count)
      return;
    output = torch::empty({count}, options);
    host_output = torch::empty({count}, options.device(torch::kCPU).pinned_memory(true));
  }
  static void CUDART_CB complete(void *p) {
    auto &s = *static_cast<Slot *>(p);
    s.flight->seconds = seconds(s.start);
    s.flight->callback_ms = diagnostic::now();
    s.flight->complete.store(true, std::memory_order_release);
    s.pending->fetch_sub(1, std::memory_order_acq_rel);
    s.pending->notify_all();
    (*s.wake)();
  }
};
bool pack_rows(Upload &u, const Bucket &b, const Rows &rows) {
  static_assert(sizeof(Row) == 75 * 8);
  if (rows.rows.size() > size_t(b.rows))
    return false;
  for (auto &r : rows.rows) {
    TORCH_CHECK((r.present >> 52) == 0 && (r.integers >> 52) == 0,
                "row presence/type bits exceed field count");
  }
  u.put(b.row_data, rows.rows.data(), rows.rows.size() * 75);
  return true;
}
int64_t capacity(int64_t n) {
  int64_t c = 1;
  while (c < n)
    c *= 2;
  return c;
}
// 根 token 已紧凑拼接；在较大图桶之间补半档，减少整段 Transformer 的容量填充。
int64_t token_bucket(int64_t n) {
  auto c = std::max<int64_t>(8, capacity(n));
  return c >= 128 && n <= c * 3 / 4 ? c * 3 / 4 : c;
}
std::shared_ptr<DeviceKernels> device_kernels(const torch::Device &device) {
  static std::mutex mutex;
  static std::map<int, std::shared_ptr<DeviceKernels>> devices;
  std::lock_guard lock(mutex);
  auto index = device.index() < 0 ? 0 : device.index();
  auto &entry = devices[index];
  if (!entry) {
    c10::cuda::CUDAGuard guard(device);
    entry = std::make_shared<DeviceKernels>();
  }
  return entry;
}
void primitive_checks(const torch::Device &device) {
  // 用可精确表示的权重对照通用算子，避免均匀策略掩盖字段或舍入错误。
  TypedProjection row_projection(32, true);
  row_projection->to(device);
  for (auto *p : {row_projection.get()}) {
    auto &w = p->bits->weight;
    auto integer =
        torch::arange(w.numel(), torch::TensorOptions().dtype(torch::kInt64).device(device));
    w.copy_((integer.remainder(7) - 3).to(torch::kBFloat16).view(w.sizes()) / 64);
  }
  DeviceKernels copies;
  auto upload = [&](const void *data, int64_t words) {
    return torch::from_blob(const_cast<void *>(data), {words}, torch::kInt64).clone().to(device);
  };
  Row a, b;
  a.ids[2] = 4097;
  a.ids[59] = -7;
  a.present = (uint64_t(1) << 51) | 1;
  a.integers = 1;
  a.values[1] = BF(0.75f).x;
  a.values[51] = BF(-0.25f).x;
  b = a;
  b.ids[2] = 4096;
  b.present = 1;
  Rows rows;
  rows.add(a);
  rows.add(b);
  rows.add(Row{});
  Inputs input;
  auto fields = rows.pack(input);
  input.upload(device);
  std::array<Field, 4> header{};
  header[0].count = rows.rows.size() * 75;
  auto metadata = upload(header.data(), 32);
  auto raw = upload(rows.rows.data(), rows.rows.size() * 75);
  auto values = torch::empty({6, 52}, bfopts().device(device));
  auto projected =
      copies.project(metadata, raw, row_projection->bits->weight, values, 32, 6, 0, true);
  auto expected = row_projection->bits(input.get(fields[0]), input.get(fields[1]));
  TORCH_CHECK(torch::equal(projected.narrow(0, 0, 3), expected),
              "raw row projection changed exact set-bit sum");
  TORCH_CHECK(torch::equal(values.narrow(0, 0, 3), input.bf(fields[2], 52)),
              "raw row changed BF16 continuous fields");
  TORCH_CHECK(projected.narrow(0, 3, 3).count_nonzero().item<int64_t>() == 0 &&
                  values.narrow(0, 3, 3).count_nonzero().item<int64_t>() == 0,
              "raw projection read capacity tail");
}
#endif
} // namespace

UpdateKernel make_update_kernel(const torch::Device &device, TargetKernel &targets) {
#ifdef HJ_CUDA
  if (device.is_cuda()) {
    c10::cuda::CUDAGuard guard(device);
    auto kernels = device_kernels(device);
    targets = [kernels](const T &mass, const T &values, const T &records, const T &blocks) {
      return kernels->targets(mass, values, records, blocks);
    };
    return [kernels](const T &g, const T &p, const std::vector<T> &state, const T &norm, double lr,
                     uint64_t step,
                     const T &loss) { return kernels->update(g, p, state, norm, lr, step, loss); };
  }
#endif
  return {};
}
CapturedUpdate make_captured_update(const torch::Device &device) {
#ifdef HJ_CUDA
  if (device.is_cuda()) {
    auto kernels = device_kernels(device);
    return [kernels](const T &g, const T &p, const std::vector<T> &state, const T &norm,
                     const T &loss, const T &control) {
      return kernels->captured_update(g, p, state, norm, loss, control);
    };
  }
#endif
  return {};
}
T typed_binary(const T &raw, bool rows) {
#ifdef HJ_CUDA
  if (raw.is_cuda())
    return device_kernels(raw.device())->binary(raw, rows);
#endif
  TORCH_CHECK(false, "raw binary projection requires CUDA");
}
std::vector<T> residual_norm_cuda(const T &a, const T &b, const T &w, const T &bias, double eps) {
#ifdef HJ_CUDA
  return device_kernels(a.device())->residual_norm(a, b, w, bias, eps);
#else
  TORCH_CHECK(false, "fused residual normalization requires CUDA");
#endif
}
T short_attention_cuda(const T &qkv, const T &cu, const T &used) {
#ifdef HJ_CUDA
  return device_kernels(qkv.device())->attention(qkv, cu, used);
#else
  TORCH_CHECK(false, "short attention requires CUDA");
#endif
}
T segmented_table(const T &data, const T &ids, const T &offsets, bool mean) {
#ifdef HJ_CUDA
  if (data.is_cuda())
    return device_kernels(data.device())->reduce(data, ids, offsets, mean);
#endif
  return std::get<0>(at::embedding_bag(data, ids, offsets, false, mean ? 1 : 0, false, {}, true));
}

struct InferenceEngine::Impl {
  Network &actor;
  torch::Device device;
  std::function<void()> wake;
  std::atomic<uint64_t> pending{0}, replays{0}, fallbacks{0}, input_bytes{0}, capacity_bytes{0},
      tokens{0}, token_capacity{0};
#ifdef HJ_CUDA
  std::shared_ptr<DeviceKernels> copies;
  std::vector<std::unique_ptr<Slot>> slots;
  void compute(Slot &s, Bucket &b) {
    copies->scatter(s.input, s.arena, b.fields, b.copy_blocks);
    auto rows = copies->project(s.input, b.row_data.tensor, actor->entity->bits->weight,
                                b.row_values.tensor, actor->width, b.rows, 0, true);
    rows = rows + actor->entity->continuous(b.row_values.tensor.view({b.rows, 52}));
    auto x = actor->encode_table(
        rows.index_select(0, b.row_ids.tensor), rows.index_select(0, b.global_ids.tensor),
        b.kinds.tensor, b.gather.tensor, b.cumulative.tensor, {b.maximum}, b.key_lengths.tensor);
    auto affine = actor->project_entities(x);
    auto context = actor->project_root_context(x.index_select(0, b.global_positions.tensor));
    copies->cache(b.descriptors.tensor, affine, context, actor->width, b.nodes);
  }
  bool prepare(Slot &s, Bucket &b, const std::vector<std::shared_ptr<Node>> &batch) {
    RECORD_FUNCTION("haojie::prepare_roots", std::vector<c10::IValue>());
    Upload upload(s.pinned);
    Rows rows;
    auto roots = roots_of(batch);
    if (roots.size() > size_t(b.candidates))
      return false;
    std::vector<int64_t> descriptors(b.candidates * 4), row_ids, global_ids, kinds, gather,
        positions;
    for (size_t i = roots.size(); i < size_t(b.candidates); ++i)
      descriptors[i * 4 + 3] = b.nodes;
    std::vector<int32_t> cumulative{0}, key_lengths;
    for (size_t i = 0; i < roots.size(); ++i) {
      auto &root = *roots[i];
      if (root.rows.size() + 1 > size_t(b.maximum))
        return false;
      positions.push_back(gather.size());
      gather.push_back(i);
      for (size_t j = 0; j < root.rows.size(); ++j) {
        gather.push_back(b.candidates + row_ids.size());
        row_ids.push_back(rows.add(root.rows[j]));
        kinds.push_back(root.kinds[j]);
      }
      cumulative.push_back(gather.size());
      key_lengths.push_back(root.rows.size() + 1);
      global_ids.push_back(rows.add(root.globals));
      s.writes.push_back({roots[i], s.used, int64_t(root.rows.size())});
      auto base = s.output.data_ptr<BF>() + s.used;
      descriptors[i * 4] = reinterpret_cast<int64_t>(base);
      descriptors[i * 4 + 1] =
          reinterpret_cast<int64_t>(base + root.rows.size() * actor->width * 2);
      s.used += (root.rows.size() * 2 + 1) * actor->width;
      descriptors[i * 4 + 2] = root.rows.size();
      descriptors[i * 4 + 3] = positions.back();
    }
    for (size_t i = roots.size(); i < size_t(b.candidates); ++i) {
      positions.push_back(gather.size());
      gather.push_back(i);
      cumulative.push_back(gather.size());
      key_lengths.push_back(1);
    }
    if (gather.size() > size_t(b.nodes) || !pack_rows(upload, b, rows))
      return false;
    gather.resize(b.nodes);
    upload.put(b.descriptors, descriptors);
    upload.put(b.global_ids, global_ids);
    upload.put(b.row_ids, row_ids);
    upload.put(b.kinds, kinds);
    upload.put(b.gather, gather);
    upload.put(b.cumulative, cumulative);
    upload.put(b.key_lengths, key_lengths);
    upload.put(b.global_positions, positions);
    upload.done();
    s.bytes = upload.next;
    s.capacity_bytes = Header;
    for (auto &f : upload.fields)
      s.capacity_bytes += f.cap * f.unit;
    b.fields = upload.fields.size();
    b.copy_blocks = upload.blocks;
    return true;
  }
  void capture(Slot &s, Bucket &b) {
    auto n = std::make_shared<Node>();
    n->root = std::make_shared<Root>();
    s.used = 0;
    TORCH_CHECK(prepare(s, b, {n}), "root capture layout failed");
    cuda_check(cudaMemcpyAsync(s.input.data_ptr(), s.pinned.data_ptr(), s.bytes,
                               cudaMemcpyHostToDevice, s.stream));
    compute(s, b);
    s.stream.synchronize();
  }
#endif
  Impl(Network &actor, const torch::Device &device, std::function<void()> wake, size_t slot_count,
       uint64_t max_roots)
      : actor(actor), device(device), wake(std::move(wake)) {
#ifdef HJ_CUDA
    if (device.is_cuda()) {
      torch::NoGradGuard guard;
      c10::cuda::getCurrentCUDAStream().synchronize();
      copies = device_kernels(device);
      auto root_capacity = capacity(std::min<uint64_t>(128, max_roots));
      // 仅为配置可用的根批捕获图；低推理上限不再预留 128 根/8192 token 的私有池。
      auto token_capacity = std::min<int64_t>(8192, root_capacity * 512);
      slots.resize(slot_count);
      for (auto &slot : slots) {
        auto &s = *(slot = std::make_unique<Slot>(device));
        c10::cuda::CUDAStreamGuard stream(s.stream);
        s.reserve((token_capacity * 2 + root_capacity) * actor->width,
                  actor->parameters()[0].options());
        s.graph_output = s.output;
        s.pending = &pending;
        s.wake = &this->wake;
        auto pool = at::cuda::graph_pool_handle();
        for (int64_t n : {8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
                          4096, 6144, 8192})
          for (int64_t c : {1, 2, 4, 8, 16, 32, 64, 128})
            if (c <= n && c <= root_capacity && n <= token_capacity) {
              auto b = std::make_unique<Bucket>(n, c, s.arena);
              capture(s, *b);
              b->graph = std::make_unique<at::cuda::CUDAGraph>();
              b->graph->capture_begin(pool);
              compute(s, *b);
              b->graph->capture_end();
              s.stream.synchronize();
              s.buckets[{n, c}] = std::move(b);
              s.writes.clear();
            }
      }
    }
#endif
  }
};
InferenceEngine::InferenceEngine(Network &actor, const torch::Device &device,
                                 std::function<void()> wake, size_t slots, uint64_t max_roots)
    : impl(std::make_unique<Impl>(actor, device, std::move(wake), slots, max_roots)) {}
InferenceEngine::~InferenceEngine() {
  wait_reads();
#ifdef HJ_CUDA
  if (impl->device.is_cuda())
    for (auto &slot : impl->slots)
      slot->stream.synchronize();
#endif
}
void InferenceEngine::wait_reads() {
  auto n = impl->pending.load(std::memory_order_acquire);
  while (n) {
    impl->pending.wait(n, std::memory_order_acquire);
    n = impl->pending.load(std::memory_order_acquire);
  }
}
std::shared_ptr<InferenceFlight>
InferenceEngine::submit(size_t slot, std::vector<std::shared_ptr<Node>> nodes, uint64_t version) {
  auto &p = *impl;
  auto flight = std::make_shared<InferenceFlight>();
  flight->nodes = std::move(nodes);
  flight->version = version;
  flight->revision = p.actor->revision;
  torch::NoGradGuard guard;
  auto start = Clock::now();
#ifdef HJ_CUDA
  if (p.device.is_cuda()) {
    auto &s = *p.slots.at(slot);
    TORCH_CHECK(!s.active, "root slot is still active");
    c10::cuda::CUDAStreamGuard stream(s.stream);
    s.active = true;
    s.flight = flight;
    s.start = start;
    s.used = 0;
    auto roots = roots_of(flight->nodes);
    auto c = capacity(roots.size());
    int64_t count = roots.size(), total = c, longest = 0;
    for (auto &r : roots) {
      count += r->rows.size();
      total += r->rows.size();
      longest = std::max<int64_t>(longest, r->rows.size() + 1);
    }
    auto n = token_bucket(total);
    auto bucket = s.buckets.find({n, c});
    // 普通回退可以扩容；图始终回到捕获时的存储，避免扩容后重放写入旧地址。
    if (bucket != s.buckets.end())
      s.output = s.graph_output;
    s.reserve((count * 2 + roots.size()) * p.actor->width, p.actor->parameters()[0].options());
    s.submitted_ms = diagnostic::now();
    s.token_count = count;
    s.timing.start();
    if (bucket != s.buckets.end() && p.prepare(s, *bucket->second, flight->nodes)) {
      cuda_check(cudaMemcpyAsync(s.input.data_ptr(), s.pinned.data_ptr(), s.bytes,
                                 cudaMemcpyHostToDevice, s.stream));
      bucket->second->graph->replay();
      p.replays++;
      p.input_bytes += s.bytes;
      p.capacity_bytes += s.capacity_bytes;
      p.token_capacity += n;
      p.actor->encoded_roots += roots.size();
      flight->graphed = true;
    } else {
      s.writes.clear();
      s.used = 0;
      p.actor->encode_roots(roots);
      for (auto &r : roots) {
        std::lock_guard lock(r->cache_mutex);
        auto entities = int64_t(r->rows.size()), width = p.actor->width;
        s.writes.push_back({r, s.used, entities});
        s.output.narrow(0, s.used, entities * width * 2)
            .view({entities, width * 2})
            .copy_(r->cached_tokens);
        s.used += entities * width * 2;
        s.output.narrow(0, s.used, width).copy_(r->cached_context);
        s.used += width;
      }
      p.token_capacity += count;
      p.fallbacks++;
    }
    // 一个槽只回送一包 BF16 数据；完成后复制为根拥有的普通 CPU 缓存。
    s.timing.compute_end();
    cuda_check(cudaMemcpyAsync(s.host_output.data_ptr(), s.output.data_ptr(), s.used * 2,
                               cudaMemcpyDeviceToHost, s.stream));
    s.timing.return_end();
    diagnostic::record("root_shape", roots.size(), count, n, longest, flight->graphed);
    p.tokens += count;
    p.pending.fetch_add(1, std::memory_order_acq_rel);
    auto status = cudaLaunchHostFunc(s.stream, &Slot::complete, &s);
    if (status != cudaSuccess) {
      s.stream.synchronize();
      p.pending.fetch_sub(1);
      p.pending.notify_all();
      s.active = false;
      cuda_check(status);
    }
    return flight;
  }
#endif
  p.actor->encode_roots(roots_of(flight->nodes));
  for (auto &root : roots_of(flight->nodes)) {
    std::lock_guard lock(root->cache_mutex);
    root->host_tokens = root->cached_tokens;
    root->host_context = root->cached_context;
    root->account_cache();
  }
  p.fallbacks++;
  flight->seconds = seconds(start);
  flight->complete = true;
  return flight;
}
void InferenceEngine::finish(const std::shared_ptr<InferenceFlight> &flight) {
  TORCH_CHECK(flight->complete.load(std::memory_order_acquire), "root slot is not complete");
#ifdef HJ_CUDA
  if (impl->device.is_cuda())
    for (auto &slot : impl->slots) {
      auto &s = *slot;
      if (s.flight != flight)
        continue;
      s.timing.report("root_batch", 0, s.submitted_ms, flight->callback_ms, flight->nodes.size(),
                      s.token_count);
      diagnostic::record("root_callback", 0, diagnostic::now() - flight->callback_ms,
                         s.timing.uncertainty_ms);
      for (auto &out : s.writes) {
        std::lock_guard lock(out.root->cache_mutex);
        auto width = impl->actor->width, count = (out.entities * 2 + 1) * width;
        auto host = torch::empty({count}, s.host_output.options().pinned_memory(false));
        std::memcpy(host.data_ptr(), s.host_output.data_ptr<BF>() + out.offset, count * 2);
        out.root->cached_tokens = T();
        out.root->cached_context = T();
        out.root->host_tokens =
            host.narrow(0, 0, out.entities * width * 2).view({out.entities, width * 2});
        out.root->host_context = host.narrow(0, out.entities * width * 2, width);
        out.root->cached_owner = impl->actor.ptr().get();
        out.root->cached_revision = flight->revision;
        out.root->account_cache();
      }
      s.writes.clear();
      s.flight.reset();
      s.active = false;
      return;
    }
#endif
}
InferenceStats InferenceEngine::stats() const {
  auto &p = *impl;
  return {p.replays.load(),        p.fallbacks.load(), p.input_bytes.load(),
          p.capacity_bytes.load(), p.tokens.load(),    p.token_capacity.load()};
}
void inference_checks(const torch::Device &device) {
  torch::NoGradGuard guard;
  cpu_policy_checks(device);
#ifdef HJ_CUDA
  if (device.is_cuda())
    primitive_checks(device);
#endif
  InputPool inputs;
  Network actor(32);
  actor->to(device);
  actor->pack_weights();
  Row row;
  row.ids[0] = 1;
  row.ids[2] = 4097;
  row.ids[8] = 7;
  row.ids[59] = -7;
  row.present = (uint64_t(1) << 51) | 1;
  row.integers = 1;
  row.values[1] = BF(0.75f).x;
  auto root = std::make_shared<Root>();
  root->game = 1;
  root->id = 1;
  root->rows = {inputs.rows.get(row), inputs.rows.get(row)};
  root->kinds = {1, 2};
  auto node = std::make_shared<Node>();
  node->root = root;
  node->context = make_context(inputs, {row}, {2, 5, 1}, {{{2}, {0, 1}}});
  node->mask = {1, 1, 0};
  node->sources = {0, -1, 1};
  node->targets = {-1, 1, 0};
  Action a{}, b{}, c{};
  a[0] = 1;
  b[2] = 4097;
  b[63] = -7;
  c[1] = 7;
  node->candidates = {inputs.actions.get(a), inputs.actions.get(b), inputs.actions.get(c)};
  auto other = std::make_shared<Node>(*node);
  other->root = std::make_shared<Root>(*root);
  other->root->game = 2;
  other->root->rows.push_back(inputs.rows.get(row));
  other->root->kinds.push_back(3);
  std::vector<std::shared_ptr<Node>> batch{node, node, other};
  auto expected = probabilities(actor->forward(batch).first, batch);
  for (auto &r : roots_of(batch))
    r->clear_cache();
#ifdef HJ_CUDA
  if (device.is_cuda())
    c10::cuda::getCurrentCUDAStream().synchronize();
#endif
  InferenceEngine engine(actor, device, [] {});
  auto old = engine.submit(0, batch, 1);
  engine.wait_reads();
  ++actor->revision;
  engine.finish(old);
  TORCH_CHECK(!root->host_valid(actor.ptr().get(), actor->revision),
              "late encoder adopted a newer revision");
  auto fresh = engine.submit(0, batch, 2);
  engine.wait_reads();
  engine.finish(fresh);
  TORCH_CHECK(root->host_valid(actor.ptr().get(), actor->revision),
              "encoder did not publish root cache");
  auto snapshot = CpuPolicy::snapshot(actor, 2);
  CpuPolicy policy;
  for (size_t i = 0; i < batch.size(); ++i) {
    auto actual = policy.score(*batch[i], snapshot);
    TORCH_CHECK(actual.size() == expected[i].size());
    for (size_t j = 0; j < actual.size(); ++j) {
      BF av, ev;
      av.x = actual[j];
      ev.x = expected[i][j];
      TORCH_CHECK(std::abs(float(av) - float(ev)) < 0.025f,
                  "native CPU policy changed full calculation");
      TORCH_CHECK(batch[i]->mask[j] || !actual[j], "CPU policy sampled masked candidate");
    }
  }
  auto wide = std::make_shared<Node>(*node);
  wide->candidates.assign(257, inputs.actions.get(a));
  wide->mask.assign(257, 1);
  wide->sources.assign(257, -1);
  wide->targets.assign(257, -1);
  TORCH_CHECK(policy.score(*wide, snapshot).size() == 257, "CPU policy truncated legal candidates");
  Network replacement(32);
  replacement->to(device);
  copy_weights(replacement, actor);
  expected = probabilities(actor->forward(batch).first, batch);
  for (auto &r : roots_of(batch))
    r->clear_cache();
#ifdef HJ_CUDA
  if (device.is_cuda())
    c10::cuda::getCurrentCUDAStream().synchronize();
#endif
  auto published = engine.submit(1, batch, 3);
  engine.wait_reads();
  engine.finish(published);
  auto next = CpuPolicy::snapshot(actor, 3);
  auto actual = policy.score(*node, next);
  for (size_t j = 0; j < actual.size(); ++j) {
    BF a, b;
    a.x = actual[j];
    b.x = expected[0][j];
    TORCH_CHECK(std::abs(float(a) - float(b)) < 0.025f, "CPU policy retained stale projections");
  }
  // 跨过旧的 512 token 边界仍走完整图；超过新边界继续核对普通计算。
  for (int extra : {600, 1100}) {
    auto large = std::make_shared<Node>(*node);
    large->root = std::make_shared<Root>(*root);
    large->root->id = 3;
    for (int i = 0; i < extra; ++i) {
      large->root->rows.push_back(inputs.rows.get(row));
      large->root->kinds.push_back(3);
    }
    expected = probabilities(actor->forward({large}).first, {large});
    large->root->clear_cache();
    auto flight = engine.submit(0, {large}, 3);
    engine.wait_reads();
    engine.finish(flight);
    TORCH_CHECK(flight->graphed == (device.is_cuda() && extra == 600),
                "long-root graph capacity or full fallback differs");
    actual = policy.score(*large, next);
    for (size_t j = 0; j < actual.size(); ++j) {
      BF a, b;
      a.x = actual[j];
      b.x = expected[0][j];
      TORCH_CHECK(std::abs(float(a) - float(b)) < 0.025f, "long-root calculation changed output");
    }
  }
  // 大批跨越旧的 16 根描述符边界，逐根核对输出，防止根之间串写。
  for (size_t count : {size_t(31), size_t(32), size_t(33), size_t(128)}) {
    std::vector<std::shared_ptr<Node>> many;
    for (size_t i = 0; i < count; ++i) {
      auto n = std::make_shared<Node>(*node);
      n->root = std::make_shared<Root>(*root);
      n->root->id = 1000 + i;
      auto varied = row;
      varied.ids[0] = int64_t(i + 1);
      n->root->rows[0] = inputs.rows.get(varied);
      many.push_back(std::move(n));
    }
    auto reference_mass = probabilities(actor->forward(many).first, many);
    for (auto &n : many)
      n->root->clear_cache();
    auto flight = engine.submit(0, many, 3);
    engine.wait_reads();
    engine.finish(flight);
    TORCH_CHECK(!device.is_cuda() || flight->graphed, "large root batch missed graph capacity");
    for (size_t i = 0; i < count; ++i) {
      auto mass = policy.score(*many[i], next);
      for (size_t j = 0; j < mass.size(); ++j) {
        BF a, b;
        a.x = mass[j];
        b.x = reference_mass[i][j];
        TORCH_CHECK(std::abs(float(a) - float(b)) < 0.025f, "large graph batch mixed root output");
      }
    }
  }
  if (device.is_cuda()) {
    InferenceEngine limited(actor, device, [] {}, 1, 1);
    for (int extra : {0, 1100, 0}) {
      auto n = std::make_shared<Node>(*node);
      n->root = std::make_shared<Root>(*root);
      n->root->rows.insert(n->root->rows.end(), extra, inputs.rows.get(row));
      n->root->kinds.insert(n->root->kinds.end(), extra, 3);
      n->root->clear_cache();
      auto reference = probabilities(actor->forward({n}).first, {n});
      n->root->clear_cache();
      auto flight = limited.submit(0, {n}, 3);
      limited.wait_reads();
      limited.finish(flight);
      TORCH_CHECK(flight->graphed == (extra == 0), "bounded inference graph capacity differs");
      auto mass = policy.score(*n, next);
      for (size_t j = 0; j < mass.size(); ++j) {
        BF a, b;
        a.x = mass[j];
        b.x = reference[0][j];
        TORCH_CHECK(std::abs(float(a) - float(b)) < 0.025f,
                    "graph fallback growth invalidated captured output");
      }
    }
  }
  std::cout << "native root slots, BF16 CPU policy, graph/full calculation, late revision, "
               "cache ownership, publication and full candidate support: passed\n";
}
} // namespace hj
