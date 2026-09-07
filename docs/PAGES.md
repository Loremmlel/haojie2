# GitHub Pages与断点续局

## 为什么原来显示README

项目原来只把构建结果写到被Git忽略的dist/index.html。Settings中选择main/(root)后，Pages发布的是Git里main根目录，而不是沙箱的dist或Actions下载产物。根目录没有index.html，因而README成为入口。Pages不会自行读取package.json然后执行任意npm脚本。

## 当前发布方式：保留main/(root)

无需切换到gh-pages分支。保持Settings → Pages → Deploy from a branch → main → /(root)。根目录index.html是构建后完整的单文件游戏，.nojekyll禁用默认Jekyll处理；它不是跳转页，也不引用dist、CDN或源码。

修改影响游戏的源码后：

```bash
npm ci
npm run check
npm test
npm run deploy
npm run test:browser
npm run test:pages
# 确认diff，再把源码与新生成的index.html/.nojekyll一并提交和推送main
```

`npm run deploy`的含义是**构建并准备发布入口**，不隐式执行git push、索取凭据或修改仓库Settings。`npm run build`仍仅生成dist/index.html，可用于离线下载。`npm run deploy:check`重新构建并逐字节检查根index是否相同；CI会拒绝“只改源码、忘记更新发布物”的提交。

原生分支发布收到main推送后会生成pages-build-deployment运行。常规CI是质量门槛，但在没有branch protection的仓库里，它不能阻止Pages先对main部署；应先在分支验证，再合并。不要手动改生成的index.html修bug。

没有使用“GITHUB_TOKEN自动提交成品后等待另一个Pages构建”链条：官方说明这种提交不触发Pages构建。本次正常main发布提交由连接器完成。未来自动化流水线应选择显式Pages artifact部署，而不是依赖机器人提交触发另一个工作流。

### 将来改成纯Actions发布

对希望完全不提交编译产物的项目，官方artifact发布更干净：将Source改为GitHub Actions，构建/验收后upload-pages-artifact + deploy-pages，并提供pages:write和id-token:write权限。当前不擅自切换设置，也不并行开启两套发布来源，避免相互覆盖。本轮优先兼容作者已经勾选的main分支发布。

## 存档在浏览器，不在Pages服务器

本游戏使用localStorage，键名固定为haojie.session.v2，JSON中保留局面、随机数和最多60步撤销/重做。UI工程版本2.0.1没有改成新存储键；不是每次重新部署就新开一个存档。

同一浏览器配置、同一origin、同一存储键、兼容的数据格式下，刷新、关闭标签、重开浏览器，以及GitHub Pages重新部署都通常可以继续。origin由协议、主机与端口确定，URL路径和Git提交SHA不参与。账号名相同的多个项目Pages页面可能共享origin，因此独立嵌入实例应使用不同storageKey。

会导致看不到原存档的主要情况是换浏览器/设备/配置、清除站点数据、隐私模式结束、存储被禁用或清理、协议/域名更换（包括启用自定义域名），或应用主动改存档schema/键名。这不是云同步。file://本地HTML的存储行为依浏览器而异，不承诺与HTTPS Pages或另一个本地路径共享；转移时导出再导入。

代码不会为了部署清空localStorage。损坏或不兼容的本机存档会暂停自动写入，避免启动新局后把原始数据悄悄覆盖；只有用户明确新建或导入有效存档后恢复写入。配额不足时尝试只保存当前局面，并提示导出完整历史；彻底写入失败会展示失败状态，不冒充“已保存”。

## 2.0到2.0.1的规则修正

旧2.0存档继续可用。唯一必要修正是旧版不限期的冲锋号令：按原抽到回合补成8回合期限，已经超过期限者移除，过去和未来快照同样修正。其余局面不重置；1.x旧规则存档仍不自动迁移。善铁及相关合成明确暂缓到3.0神龛模式，不影响现在的炎魔之心配方。

## 官方参考

- [Pages入口文件](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
- [配置发布来源及GITHUB_TOKEN注意事项](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
- [自定义Pages工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
- [MDN localStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)

## 2.1人机模式

同一单文件产物现在内联AI Worker及分片降级代码；页面运行不下载模型/Worker资产，不新增后端。模式、难度和人类阵营保存在Session.match，继续使用haojie.session.v2。旧v2局面缺少设置时作为同屏双人恢复。人机悔棋同时撤销电脑回应，并可重新做回实际历史；AI思考线程不在存档中，刷新后按公开当前局面重新规划。

发布前额外运行npm run test:browser:ai；CI仍校验生成的index.html，源码更新时不要漏更新发布物。减少动态效果或关闭音效不会关闭AI，也不会改变规则概率。
