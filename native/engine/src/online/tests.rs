//! 小树保留正式规则提交，只注入候选结构以覆盖失败、回溯和实际概率。
use super::*;
use crate::model::{Command, Point};
use crate::tree::{Choice, ChoiceKey, Node};

#[test]
fn small_tree_failure_backtrack_and_retry_keep_probability_and_rng() {
    let cat = catalog::embedded().unwrap();
    let mut state = runtime::create(73127, false, &cat).unwrap();
    state.phase = "play".into();
    state.summon_slots = 0.0;
    state.turns.insert("1".into(), 3.0);
    state.turns.insert("2".into(), 3.0);
    let mut unit =
        crate::resolution::template(&state, "26", 1, Point { x: 4.0, y: 4.0 }, "unit", &cat);
    unit.born = 0.0;
    state.units = vec![unit];
    let original = serde_json::to_value(&state).unwrap();
    let mut game = Game {
        saved: SavedGame {
            id: 1,
            state,
            random: 1,
            learner: 1,
            opponent: 0,
            root_id: 0,
            commands: 0,
            microsteps: 0,
            failures: 0,
            backtracks: 0,
            offer_interrupt: false,
            search: None,
            reported_terminal: false,
            waiting: false,
        },
        tree: None,
        announced: false,
    };
    game.start_root(&cat).unwrap();
    let tree = game.tree.as_mut().unwrap();
    let branch = tree
        .node(&[])
        .unwrap()
        .choices
        .iter()
        .find(|choice| choice.next.is_some())
        .unwrap()
        .clone();
    let choice = |name, command, status| Choice {
        key: ChoiceKey::Static(name),
        command: Command::new(command),
        status,
        subject: None,
        next: None,
    };
    tree.inject_node(Node {
        cursor: vec![],
        prefix: None,
        stage: "action",
        choices: vec![
            branch,
            choice("invalid", "attack", "uncertain"),
            choice("end-a", "end", "available"),
            choice("end-b", "end", "available"),
        ],
    });
    tree.inject_node(Node {
        cursor: vec![0],
        prefix: None,
        stage: "unit",
        choices: vec![],
    });
    // 从已选中空子树的边界开始；回溯后必须恢复根并永久排除该子树。
    game.saved.search.as_mut().unwrap().cursor = vec![0];
    game.advance(None, &cat).unwrap();
    assert_eq!(game.saved.backtracks, 1);
    assert_eq!(
        game.saved.search.as_ref().unwrap().excluded[&vec![]],
        vec![0]
    );
    let reply = game.advance(Some(vec![0.0, 1.0, 3.0, 4.0]), &cat).unwrap();
    assert_eq!(i32::from_le_bytes(reply[72..76].try_into().unwrap()), 1);
    assert!(
        (f64::from_le_bytes(reply[76..84].try_into().unwrap()) - (1.0f64 / 8.0).ln()).abs() < 1e-12
    );
    assert_eq!(game.saved.failures, 1);
    assert_eq!(game.saved.microsteps, 1);
    assert_eq!(serde_json::to_value(&game.saved.state).unwrap(), original);
    assert_eq!(
        game.saved.search.as_ref().unwrap().excluded[&vec![]],
        vec![0, 1]
    );
    assert_eq!(game.saved.search.as_ref().unwrap().sent_events, 3);
    let mut random = Random::new(1);
    random.next();
    assert_eq!(game.saved.random, random.0);
    let reply = game.advance(Some(vec![0.0, 0.0, 1.0, 3.0]), &cat).unwrap();
    let selected = i32::from_le_bytes(reply[72..76].try_into().unwrap());
    assert!([2, 3].contains(&selected));
    let expected = if selected == 2 { 0.25f64 } else { 0.75f64 };
    assert!((f64::from_le_bytes(reply[76..84].try_into().unwrap()) - expected.ln()).abs() < 1e-12);
    assert_eq!(game.saved.microsteps, 2);
    assert!(game.saved.commands >= 1);
    assert!(game.saved.root_id > 1);
}
