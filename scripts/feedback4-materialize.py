#!/usr/bin/env python3
"""One-shot reviewed source edits on the isolated feedback4 branch. Removed before final commit."""
from pathlib import Path
import subprocess
root = Path('.')
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'fix/feedback4-20260921'
subprocess.run(['git', 'merge-base', '--is-ancestor', '7824f47c0deb0797819833f6641021f87f4aecbd', 'HEAD'], check=True)
def edit(path, old, new, count=1):
    p = root/path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:80], s.count(old), count)
    p.write_text(s.replace(old, new))
edit('src/engine/traits.ts', '// Printed equipment restrictions survive silence; inherited restrictions apply as abilities.', '''// Printed giant restrictions also apply to inherited giant traits and survive silence.
export const refusesConversion = (u: Unit): boolean => hasTrait(u, 5);
export const isHookImmune = (u: Unit): boolean => hasTrait(u, 5);
// Printed equipment restrictions survive silence; inherited restrictions apply as abilities.''')
edit('src/engine/abilities.ts', '  refusesWeapons,', '  refusesWeapons,\n  refusesConversion,\n  isHookImmune,')
edit('src/engine/abilities.ts', '(t) => ring(u, t.unit ?? t) && (t.owner === u.owner || topTarget(s, t)),', '''(t) =>
          (t.unit ? allegiance(s, t.unit) : t.owner) !== u.owner &&
          ring(u, t.unit ?? t) &&
          topTarget(s, t),''')
edit('src/engine/abilities.ts', '''      const t = enemy(c.targetId),
        to = point(c.x, c.y);
      ensure(''', '''      const t = enemy(c.targetId),
        to = point(c.x, c.y);
      ensure(!isHookImmune(t.unit!), '大肉比不能被钩子牵引。');
      ensure(''')
edit('src/engine/abilities.ts', "      ensure(canPlace(s, v, to), '身前没有合法落位。');", "      ensure(!isHookImmune(v), '大肉比不能被钩子牵引。');\n      ensure(canPlace(s, v, to), '身前没有合法落位。');")
edit('src/engine/abilities.ts', "  if (card.kind === 'u26') ensure(target?.unit, '心灵之火只能选择随从。');", "  if (card.kind === 22) ensure(!refusesConversion(target!.unit!), '大肉比不能使用策反。');\n  if (card.kind === 'u26') ensure(target?.unit, '心灵之火只能选择随从。');")
edit('src/engine/combat.ts', '  signedAttack,', '  signedAttack,\n  refusesConversion,\n  isHookImmune,')
edit('src/engine/combat.ts', '''        convert &&
        victim &&''', '''        convert &&
        !refusesConversion(u) &&
        victim &&''')
edit('src/engine/combat.ts', '''        !isShrine(victim) &&
        !hasTrait(victim, 5) &&''', '''        !isShrine(victim) &&''')
edit('src/engine/combat.ts', '    if (conversion && attackLoss > 0) {', '    if (conversion && !refusesConversion(u) && attackLoss > 0) {')
edit('src/engine/combat.ts', '''      if (hasTrait(victim, 5))
        emit(s, {
          type: 'shield',
          to: victim,
          owner: victim.owner,
          action: 'conversion',
          stage: 'blocked',
          text: '大肉比 · 无法策反',
        });
      if (!hasTrait(victim, 5) && !protectedEffect(s, t, skillSource, ctx)) {''', '''      if (!protectedEffect(s, t, skillSource, ctx)) {''')
edit('src/engine/combat.ts', "    if (hasTrait(u, 'formless') && passive(s, u) && alive(s, u))", "    if (hasTrait(u, 'formless') && passive(s, u) && alive(s, u) && !isHookImmune(victim))")
edit('src/engine/reactions.ts', "import { hasTrait } from './traits';", "import { hasTrait, isHookImmune } from './traits';")
edit('src/engine/reactions.ts', '''    !victim ||
    source.owner''', '''    !victim ||
    isHookImmune(victim) ||
    source.owner''')
edit('src/ai/threats.ts', '  abilityKinds,', '  refusesConversion,\n  abilityKinds,')
edit('src/ai/threats.ts', '  if (!effect) return result;', "  if (!effect || (type === 'convert' && refusesConversion(u))) return result;")
edit('src/ai/threats.ts', "    if (type === 'convert' && hasTrait(victim, 5)) reason = '大肉比无法被策反';\n    else if (has(view, victim, 'immune'))", "    if (has(view, victim, 'immune'))")
edit('src/engine/index.ts', '  canDeployKind,\n} from', '  canDeployKind,\n  isHookImmune,\n} from')
edit('src/ui/game/selection.ts', '  asTarget,', '  asTarget,\n  isHookImmune,')
edit('src/ui/game/selection.ts', "    if (i.action.id === 'sacrifice'", "    if (i.action.id.split(':')[0] === 'hook' && t.unit && isHookImmune(t.unit)) return false;\n    if (i.action.id === 'sacrifice'")
edit('src/engine/options.ts', "label: '献祭 2/2：选择射击列'", "label: '献祭 2/2：选择射击列，可命中敌方基地'")
edit('src/engine/player-view.ts', "'3.0-feedback3' as const", "'3.0-feedback4' as const")
edit('src/engine/catalog.ts', "RULESET_ID = '3.0-feedback3-2026-09-20'", "RULESET_ID = '3.0-feedback4-2026-09-21'")
edit('src/engine/catalog.ts', '体型外围一圈12格（含四个对角）内的目标造成15伤害，包含友方，按每个覆盖格分别结算。不能被策反。', '体型外围一圈12格（含四个对角）内的敌方目标造成15伤害，不伤害友方，按每个覆盖格分别结算。不能使用策反，但可以被策反；不能被任何钩子牵引。')
edit('src/engine/catalog.ts', '沿进攻方向对射程内该列第一个敌方造成', '沿进攻方向对射程内该列第一个敌方（含基地）造成')
edit('src/engine/catalog.ts', '金身免疫；大肉比不能被策反，基地也不能被策反。', '金身免疫；不能对友方大肉比施放此法术，但敌方大肉比可以被策反。基地不能被策反。')
edit('src/ui/library/Rules.tsx', '大肉比不能被策反，普通攻击10、周围一圈技能伤害15。对其他目标，策反先结算伤害，', '大肉比不能使用策反，但可以被策反；不能被任何钩子牵引。普通攻击10、周围一圈技能对敌伤害15，不伤友方。策反先结算伤害，')
edit('docs/RULES.md', '大肉比5不能被策反（沉默也不解除这一限制）；其普通攻击10，外围一圈技能伤害15，2×2与半速蓄力移动不变。其他目标策反必须', '大肉比5不能使用22策反，但可以被策反；不能被普通钩子、超级钩子、无相勾等任何钩子牵引（沉默不解除禁手；普通击退不属于钩子）。其普通攻击10，外围一圈技能仅对敌方造成每覆盖格15伤害，不误伤友方棋子或己方基地，2×2与半速蓄力移动不变。策反必须')
edit('docs/RULES.md', '对所选范围内一列的第一个敌方造成伤害；', '对所选范围内一列的第一个敌方（包括敌方基地）造成伤害；')
edit('docs/RULES.md', '# 浩劫3.0规则契约\n', '# 浩劫3.0规则契约\n\n2026-09-21反馈4修订见`FEEDBACK-2026-09-21.md`；大肉比的友伤、策反方向与钩子限制以该次作者更正为准。\n')
edit('tests/ultimate/synthesis.test.ts', '    big = add(mirror, 5, 1, 3, 3);', "    big = add(mirror, 'u4', 1, 3, 3);\n  big.size = 2; // A BW-expanded non-giant still uses full 2x2 pull geometry.")
edit('tests/ai/feedback.test.ts', 'conversion excludes giant', 'conversion includes enemy giant')
edit('tests/ai/feedback.test.ts', '  assert.equal(analysis.targets.find((t) => t.id === giant.id)?.probability, 0);', '  assert.ok(analysis.targets.find((t) => t.id === giant.id)!.probability > 0);')
edit('tests/core/feedback.test.ts', 'conversion never changes its side', 'conversion can change its side')
edit('tests/core/feedback.test.ts', 'assert.equal(unit(n, big.id).owner, 2);', 'assert.equal(unit(n, big.id).owner, 1);')
edit('tests/core/rules.test.ts', 'and can hurt allies', 'and spares allies')
edit('tests/core/rules.test.ts', 'assert.equal(unit(s, friend.id).hp, 35);', 'assert.equal(unit(s, friend.id).hp, 50);')
edit('package.json', '"test:browser:online":', '"test:browser:feedback4": "node tests/browser/feedback4.mjs",\n    "test:browser:online":')
edit('.github/workflows/ci.yml', '      - run: npm run test:browser:feedback3', '      - run: npm run test:browser:feedback3\n      - run: npm run test:browser:feedback4')
edit('.github/workflows/ci.yml', 'run: npm run play:cli -- --replay docs/playtests/cli-feedback3-20260920.jsonl', '''run: npm run play:cli -- --replay docs/playtests/cli-feedback4-20260921.jsonl
      - name: Check out the September 20 feedback3 rules
        uses: actions/checkout@v4
        with:
          ref: 7824f47c0deb0797819833f6641021f87f4aecbd
          path: artifacts/september20-rules
          persist-credentials: false
      - name: Verify the unmodified September 20 recording
        run: |
          ln -s "$GITHUB_WORKSPACE/node_modules" artifacts/september20-rules/node_modules
          npm --prefix artifacts/september20-rules run play:cli -- --replay "$GITHUB_WORKSPACE/docs/playtests/cli-feedback3-20260920.jsonl"''')
p = root/'AGENTS.md'
p.write_text(p.read_text() + '''

## 2026-09-21 反馈4不变量

见`docs/FEEDBACK-2026-09-21.md`。普通5技能无友伤；禁手是不能使用策反，不是不能被策反。所有钩子共用`isHookImmune`，包括继承能力和旧档未决牵引；沉默不解除，普通击退不套钩子免疫。被巨大化的非5棋子仍可牵引。部署行保持回合开始快照：人数差≥2；不在回合中或存档恢复时重算。当前CLI反馈4录制使用新规则，反馈3录制固定7824f47c验证，禁止改写旧指纹。联网断点续玩保存在宿主服务器的完整GameState，公开PlayerView不是存档；恢复不得重开回合或泄露私有数据。
''')
p = root/'docs/ONLINE-ADAPTATION.md'
p.write_text(p.read_text() + '''

## 保存与断点续玩（2026-09-21）

**普通断线重连、退出后继续未结束对局，网站适配即可，不要求再重构游戏引擎。** GameState本身已保存完整局面、seed/rng/serial、手牌、技能状态、未决反应、部署行快照、神龛秘密选择及局部时钟快照。`tests/session/online-resume.test.ts`验证JSON往返后继续相同命令得到相同局面与随机状态；不是只保存棋盘截图或初始种子。

宿主应在服务器持久化一个包含完整GameState的存档封套，并保存规则标识、准确源代码提交、房间ID、双方认证席位、状态、单调递增revision及请求去重结果。已有每步原子持久化时，“保存”通常只需暂停/保留房间，不必再造一份引擎存档。

恢复时从持久化存档取回权威状态，检查版本与数据结构；重新鉴权席位，再分别调用`getPlayerView`，用`kind: 'snapshot'`交给已有`HaojieOnlineGame`。禁止把完整存档、seed/rng、另一方未揭示的选择发给浏览器，也不能把PlayerView补随机数后当作权威存档。不要在恢复时调用`createGame`重抽、重新执行回合开始、重新计算部署行或清空待处理反应。

一次成功命令的局面、revision与去重回执必须原子保存后再确认成功；重连重试沿用原requestId。连接断开不应销毁未结束房间。暂停是否须双方同意、房间保留期限、恢复按钮、列表、身份权限和并发控制均属网站职责。本次没有检查或修改网站数据库，不能据此保证生产房间已自动留存。

“继续最新进度”与“双方回到较早检查点”是两项功能：后者需要网站做双方同意与回退操作协议，不能允许单方上传任意局面。恢复旧检查点时不倒退同一房间revision；可在新revision安装旧局面快照，或创建新matchId并重新建立会话，同时处理旧的在途请求和去重范围。

活动房间必须继续用保存时对应规则/代码，或走明确、经过验证的迁移。此轮规则从feedback3变为feedback4，不能偷偷热替换正在进行的旧局。双端代码与规则标识一起固定；同为v2存档不代表规则完全相同。无需为本功能新增WS、把联网存档放进localStorage，或向纯引擎加入数据库依赖。
''')
Path('.github/workflows/feedback4-materialize.yml').unlink()
Path(__file__).unlink()
print('Applied reviewed source edits; removed the temporary branch-only materializer.')
