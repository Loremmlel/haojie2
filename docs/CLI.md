# 命令行对局与可复现复盘

不需要浏览器、服务器或模型API。CLI与页面共用`applyCommand`、存档格式、AI候选/评分/搜索，以及分配搜索预算的函数；并非另写一套简化规则。

```bash
npm ci
npm run play:cli -- --new --seed 20260907 --human 1 --difficulty hard --save artifacts/my-match.json
```

可执先或后；难度为`easy`、`medium`、`hard`。`--new`明确创建新局并替换指定存档/回放；不带它会读取已有的`--save`。使用`--load`可载入网页导出的JSON。不要同时用浏览器或两个CLI进程改写同一存档。

## 操作

`show`查看9×13棋盘、棋子ID、阵营、坐标、生命、攻击、射程、剩余攻击、操作模式、蓄力和手牌。棋盘上的101表示P1的第01项棋子，真正命令使用列表里的稳定ID，例如`u7`；基地ID为`base-1`/`base-2`。

```text
summon                       普通召唤一次
summon ultimate              支付人头进行终极召唤
begin                        完成抽取后进入行动阶段
play中的例子：
deploy c2 5 7                 部署手牌c2
move u7 5 8                   移动单位u7
attack u7 u9                  攻击或治疗目标
charge u7 attack              为攻击蓄力（也支持move/skill）
cast c20 u7                   对u7施放法术
equip c21 u7                  装备
finish u7                     提前结束该单位的连续攻击/移动模式
react u9                      结算死亡反应或伤害转化
react 5 6                     弹出或小屋召唤落点
end                          结束回合
```

`actions ID`显示引擎描述的能力和所需选点；`legal [ID]`列出经过验证的AI候选。**候选清单并不等于所有合法动作**，完整命令始终可以直接输入JSON，例如：

```json
{ "type": "skill", "unitId": "u7", "targetId": "u9", "x": 5, "y": 6 }
```

`go`让AI行动至下一人类决策点，`step`只执行一条AI命令。AI主动回合中触发人类奶妈死亡反应时，`go`会停下来；在人类回合中触发AI反应，仍需用`go`/`step`让其处理。不会抢走反应归属。`quit`退出；每步成功操作已保存，非法操作不写入半个结果。

无交互环境可逐条执行：

```bash
npm run play:cli -- --save artifacts/my-match.json --command 'show'
npm run play:cli -- --save artifacts/my-match.json --command 'go'
```

## 搜索与记录

CLI没有人为的观战延时，但AI仍使用与页面相同的回合共享计算预算。所有思考只读取公开Observation，不读真实PRNG。`--nodes 1000 --ms 100000`可以显式覆盖每决策预算，用于可重复的固定节点实验；这不是默认网页速度。机器负载和墙钟预算会使具体方案有差别，不保证两次限时运行选择逐字相同。

存档旁会写入`.jsonl`回放。记录包含初始Session、每条实际命令、操作方、前后公开指纹、事件及可选评分trace；初始Session有真实随机状态是为了回放，与送给AI的Observation分开。trace列出候选顺序、分值分项、概率分支、策反/处决承载者和可达目标；“静态期望”不是已经发生的伤害，“存活折扣”是启发式，不是测得概率。随机结果出现后仍以真实局面重算。

```bash
npm run play:cli -- --replay artifacts/my-match.jsonl
```

回放只执行记录中的命令，不重新调用AI；每一步校验前后指纹。不兼容的命令、被篡改的记录、与存档不一致的旧回放均明确报错，不覆盖旧记录。也支持读取仓库中压缩的`.jsonl.gz`实战记录。

`docs/playtests/cli-human-20260907.jsonl.gz`是开发中人工操作P1、与困难AI交手的记录，共214条实际指令（82条人工、132条AI），停在第11个己方回合，基地295/225，未决。压缩版保留命令、事件及校验指纹，省略冗长的候选分值；交互CLI的新记录仍提供候选trace。它用于定位决策问题，期间有代码修正和人工失误，**不计入冻结版本胜率比较**。

## 与冻结旧版本比较

把可信旧源码中的`src/ai/index.ts`用esbuild打包成Node ESM，显式提供该文件；不要导入不可信的JavaScript。

```bash
# 从旧工作树构建；366bfe1与02db46e的AI代码相同。
./node_modules/.bin/esbuild ../old-haojie/src/ai/index.ts --bundle --platform=node --format=esm --outfile=artifacts/baseline.mjs
npm run bench:ai:compare -- /absolute/path/to/artifacts/baseline.mjs
```

比较默认种子7、42、20260907，并交换先后手。可用环境变量`AI_NODES`、`AI_PLIES`、`AI_SEEDS`、`AI_REPORT`调整计算节点、对局上限和报告路径；`AI_LEVELS=hard,hard`指定新旧难度，也可比较两边medium/easy。所有结果记录源码摘要、节点预算、真正胜者、未决状态和实际回放。达到上限一律记未决，不能按基地血量替代规则判胜。

同种子不意味着两种策略每回合永远抽相同牌：不同的暴击、死亡和改判会消耗不同次数的真实随机数。这是原游戏规则，不额外修改它来制造测试优势。

少量固定种子不是Elo，也不能证明与规则作者五五开。比较工具的作用是发现退步、行为异常和明显策略差距，之后再扩大独立种子组与真实人类对局。
