# 浩劫 2.0 · haojie2

React + TypeScript 同屏双人回合制战棋。**生产构建只有一个 `dist/index.html`**：React、样式、图标、棋子和特效全部内联，运行时无 CDN、外部字体、图片或 API 请求。游戏引擎不依赖 React、DOM、网络、本地存储或系统时间。

规则依据本次提供的 [《浩劫.docx》](docs/source/浩劫.docx)，原文件和逐段文本均已归档。项目显示名称为 **浩劫**，仓库与包名保持 `haojie2`，不更改现有仓库地址。

## 试玩与构建

开发需要 Node.js 22：

```sh
npm ci
npm run dev
```

开发服务默认 `http://127.0.0.1:4173`，源码修改后自动重建，手动刷新浏览器。生成离线成品：

```sh
npm run build
```

将 `dist/index.html` 复制到任意目录并在现代浏览器打开即可；运行游戏不需要 Node.js。也可把该文件作为现有网站的静态页面。受管理的浏览器可能禁止 `file://`，这种情况应通过网站访问，而不是绕过浏览器策略。

正式对局从空棋盘开始，双方基地300生命，苍穹方先手。**先选择普通或终极召唤，完成所有召唤后点“完成召唤，开始行动”。**“新对局 → 载入演示棋局”可直接体验装备、法师、叠放、伤害和悔棋。双方轮流操作同一设备，手牌公开。

## 本次更新

| 模块       | 实现                                                                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 原规则修正 | 第四属性改为攻击次数；每回合选择一种操作模式；主动蓄力；名刀可蓄力移动；大肉比周围震地；金身拦截敌方技能、法术、死吧！和策反；普通20改为新版死亡判定。                      |
| 终极召唤   | 击杀人头、2人头兑换、揭牌前支付、28编号独立召唤池、SZF变体、改判、金色编号及召唤特效。                                                                                      |
| 装备       | 寒冰法杖、弑君、靴子、炎魔之心；储存期限、职业限制、替换装备、死亡清除、炎魔之王合成。                                                                                      |
| 终极能力   | 沉默/眩晕、冰冻中立、灼烧、十字攻击、巨大化、吸血成长、持续火线、冲撞弹出、虹吸、免疫塔、单体回合推进、反击、复活、击退、小屋死亡召唤、延迟标记、克隆叠放、心灵之火、矿工。 |
| 交互与图鉴 | 多步选目标、复活记录选择、克隆逐个控制、地面标记、火线、虹吸引导线、护盾/冰冻/装备标识；59个图鉴条目，支持普通/终极/法术/武器/变体筛选与能力搜索。                          |
| 历史和保存 | 最近60步完整悔棋/重做；人头、随机数、装备、连锁反应、叠放批次、阵亡记录和独立时钟全部可恢复；自动保存及JSON导出/导入。                                                      |

**不是把“攻击两次”改成“可自由行动两次”。** 射手选攻击后能攻击两次，但不能攻击一次再移动。矿工、靴子、免费技能及冲锋号令是文档列明的例外。引擎负责这些限制，界面只提交命令。

## 与旧版相比

完整逐条对照在 [变更核对表](docs/CHANGELOG-2.0.md)。旧版的曼哈顿距离、3′人数算法、己方回合储存期限、行控制快照、投石机基地标记及确定性悔棋得到文档确认，继续保留；与新回复冲突的旧实现已经修改。

**旧1.0存档不自动迁移。** 第四属性和操作规则变更会改变旧局面的含义，不能只加字段后假装兼容。2.0使用独立的 `haojie.session.v2` 本地存储键，旧键不删除；导入旧档时明确拒绝并说明原因，原文件不被改写。

## 2.0.1工程更新与作者补充

冲锋号令储存期限已确认**8回合**；善铁和相关合成属于暂缓的**3.0神龛模式**，不再作为当前2.x未完成项。三张炎魔之心合成炎魔之王维持原规则。

Game.tsx改为页面组合根；组件、存档、控制流程和样式按功能拆分。保留原生CSS/SVG，不增加组件库或动画库；详细成本收益和目录归属见[工程说明](docs/ENGINEERING.md)。

### GitHub Pages

保持Settings → Pages → Deploy from a branch → **main / (root)**。根index.html是完整游戏发布物，.nojekyll禁用Jekyll，不再以README作首页。更新源码后运行：

```bash
npm run deploy       # 生成dist/index.html，并准备根目录的发布入口
npm run deploy:check # 验证根入口与源码构建逐字节一致，CI也会检查
```

随后把源码与index.html一同提交推送；Pages不会自动运行package.json里的脚本。详情见[发布与存档](docs/PAGES.md)。

**重新部署不会主动丢存档。** 同一浏览器、同一origin及兼容的schema继续读`haojie.session.v2`。不是云存档；换设备/域名、清除站点数据等仍需导入导出。旧2.0号令卡按抽取回合补成8回合期限，撤销历史一并兼容修正，不更换存储键。

## 嵌入现有网站

整个组件：

```tsx
'use client'; // Next.js App Router适用
import { HaojieGame } from './haojie/src';

export default function GamePage() {
  return <HaojieGame storageKey="my-site.haojie.match" />;
}
```

只复用规则引擎：

```ts
import { createGame, applyCommand } from './haojie/src/engine';
let state = createGame(20260907);
state = applyCommand(state, { type: 'summon', ultimate: false });
```

React入口支持 `initialState`、`storageKey={null}`、`onStateChange`。样式以 `.hj-game` 限定，独立页面对 `body` 的重置不进入组件。详见 [架构与接入](docs/ARCHITECTURE.md)。当前没有联网房间、AI对手或服务端，未来可复用引擎并另行建立权威服务端。

## 验证与维护

```sh
npm run format:check
npm run check
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

测试按 `core`、`ultimate`、`session` 分类，每种棋子有稳定的行为断言，不依赖React内部实现和像素快照。浏览器测试从生产HTML开始，执行真实按钮操作，包括装备、法术、复活、虹吸、冲撞、改判、克隆控制、胜负、悔棋和手机布局。

默认浏览器验收使用 **网络离线的 `file://` 页面**。特定沙箱禁用页面导航时，可使用 `HAOJIE_RENDER_ONLY=1 npm run test:browser` 在内存里渲染同一成品，报告明确标记 `in-memory-render-only`，不把它冒充本地文件和自动存档验收。GitHub CI仍运行默认完整模式。

CI产物：`haojie2-single-html`、`source-snapshot`、`acceptance-evidence`。部署动画、弹道和音效不会驱动规则计时；音效默认关闭，界面遵守系统“减少动态效果”设置。

运行时依赖仍只有React与ReactDOM；构建用esbuild，规则测试用Node测试运行器+tsx，浏览器验收用Playwright。所有依赖已固定在lockfile，第三方运行时许可包含在发行HTML里。

HTTP同源续局与多实例嵌入验收：`npm run test:pages`。组件与样式定位见`docs/ENGINEERING.md`；原始规则文档保留在`docs/source/`。
