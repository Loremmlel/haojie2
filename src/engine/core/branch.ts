import type { GamePosition } from '../types';

// 与 Rust State 的拥有型核心字段对应；其余对象字段按顶层 JSON 子树共享。
const ownedFields = new Set(['turns', 'pending', 'siphons', 'events', 'deployRows', 'bases']);
const commits = new WeakMap<object, () => GamePosition>();

/** 同一只读局面的来源图只登记一次；每次分支仍拥有独立代理/写入表，不能跨局面复用。 */
export function createPositionFork<S extends GamePosition>(position: S): () => S {
  let sources: WeakSet<object> | undefined;
  return () => forkPosition(position, false, (sources ??= sourceObjects(position)));
}

/** 收口同步草稿：剥离代理，未写入的子树保留只读共享；外部出口再导出独立快照。 */
export function commitPosition<S extends GamePosition>(position: S): S {
  const commit = commits.get(position);
  if (!commit) return position;
  const result = commit() as S;
  commits.delete(position);
  return result;
}

/**
 * 双端共同算法：复制根和实体引用列表；实体/扩展字段首次写入时复制整棵子树。
 * 分支只在同步结算内存在；正式成功由入口导出独立纯数据，预检/失败直接丢弃。
 * 输入是引擎维护的纯数据局面；来源表区分输入对象和分支中新建的对象，不设单端旧算法回退。
 */
export function forkPosition<S extends GamePosition>(
  position: S,
  willCommit = false,
  sources = sourceObjects(position),
): S {
  type Owner = { root?: Node };
  type Node = { data: any; proxy: any; owner: Owner; owned: boolean };
  const nodes = new WeakMap<object, Node>();

  const nodeFor = (value: any, owner: Owner): Node => {
    const known = nodes.get(value);
    if (known) return known;
    const node: Node = { data: value, proxy: undefined, owner, owned: false };
    const array = Array.isArray(value);
    const target = array ? [] : {};
    node.proxy = new Proxy(target, {
      get(_target, key) {
        return wrap(Reflect.get(node.data, key), owner);
      },
      set(_target, key, value) {
        writable(node);
        return Reflect.set(node.data, key, value);
      },
      deleteProperty(_target, key) {
        if (!Object.hasOwn(node.data, key)) return true;
        writable(node);
        return Reflect.deleteProperty(node.data, key);
      },
      has(_target, key) {
        return Reflect.has(node.data, key);
      },
      ownKeys() {
        return Reflect.ownKeys(node.data);
      },
      getOwnPropertyDescriptor(_target, key) {
        const descriptor = Reflect.getOwnPropertyDescriptor(node.data, key);
        if (!descriptor) return undefined;
        if (array && key === 'length')
          return { ...descriptor, writable: true, configurable: false };
        return {
          ...descriptor,
          value: wrap(descriptor.value, owner),
          writable: true,
          configurable: true,
        };
      },
      defineProperty(_target, key, descriptor) {
        writable(node);
        return Reflect.defineProperty(node.data, key, descriptor);
      },
    });
    nodes.set(value, node);
    nodes.set(node.proxy, node);
    return node;
  };
  // 新插入对象属于当前分支，必须保持原引用，不能把规则保存的 e 变成另一份代理身份。
  const wrap = (value: any, owner: Owner): any =>
    value && typeof value === 'object' && (sources.has(value) || nodes.has(value))
      ? nodeFor(value, owner).proxy
      : value;
  const detach = (root: Node) => {
    const copies = new Map<Node, any>();
    const copy = (node: Node): any => {
      if (copies.has(node)) return copies.get(node);
      const source = node.data;
      const result: any = Array.isArray(source) ? source.slice() : { ...source };
      copies.set(node, result);
      for (const key of Object.keys(result)) {
        const value = result[key];
        if (value && typeof value === 'object' && (sources.has(value) || nodes.has(value)))
          result[key] = copy(nodeFor(value, node.owner));
      }
      return result;
    };
    copy(root);
    for (const [node, data] of copies) {
      node.data = data;
      node.owned = true;
      nodes.set(data, node);
    }
  };
  const writable = (node: Node) => {
    if (node.owned) return;
    detach(node.owner.root!);
    // 规则可能把已有来源对象放入已分离实体；写入该对象前仍需隔离。
    if (!node.owned) detach(node);
  };
  const shared = (value: any, owned = false) => {
    if (!value || typeof value !== 'object') return value;
    const owner: Owner = {};
    owner.root = nodeFor(value, owner);
    if (owned) detach(owner.root);
    return owner.root.proxy;
  };
  const result: any = { ...position };
  for (const key of Object.keys(result)) {
    const value = result[key];
    if (key === 'units' || key === 'landmarks')
      result[key] = value?.map((unit: any) => shared(unit));
    else if (ownedFields.has(key)) result[key] = shared(value, true);
    else result[key] = shared(value);
  }
  if (willCommit)
    commits.set(result, () => {
      const seen = new Map<object, any>();
      const materialize = (value: any): any => {
        if (!value || typeof value !== 'object') return value;
        const node = nodes.get(value);
        // 未分离的整个复制单元从未被写入，原输入是纯数据，不需要遍历其冷字段。
        if (node && !node.owned) return node.data;
        if (!node && sources.has(value)) return value;
        const data = node?.data ?? value;
        if (seen.has(data)) return seen.get(data);
        seen.set(data, data);
        for (const key of Object.keys(data)) data[key] = materialize(data[key]);
        return data;
      };
      return materialize(result);
    });
  return result;
}

/** JS 没有 Rust 的静态所有权信息；只登记来源，不复制对象或改变分支算法。 */
function sourceObjects(position: GamePosition): WeakSet<object> {
  const sources = new WeakSet<object>();
  const visit = (value: any) => {
    if (!value || typeof value !== 'object' || sources.has(value)) return;
    sources.add(value);
    for (const key of Object.keys(value)) visit(value[key]);
  };
  visit(position);
  return sources;
}
