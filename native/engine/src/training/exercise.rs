//! 可解析短环境，仅供学习验收；复用正式 MC 选择、回溯和决策出口，不是浩劫棋力结果。
use crate::{
    encoding,
    model::Command,
    policy::Random,
    sampler::{self, Inference, McChoice, McNode, Selected},
};
use serde_json::{Value, json};

pub fn run(request: &Value, inference: &mut dyn Inference) -> Result<Value, String> {
    let case = request["case"]
        .as_u64()
        .filter(|v| *v < 3)
        .ok_or("invalid exercise case")? as usize;
    let seed = request["samplerSeed"]
        .as_u64()
        .ok_or("missing exercise seed")? as u32;
    let outcome_seed = request["outcomeSeed"]
        .as_u64()
        .ok_or("missing outcome seed")? as u32;
    let mut random = Random::new(seed);
    let mut nodes = |path: &[usize]| {
        let count = if path.is_empty() {
            3
        } else if path[0] == 0 {
            0
        } else {
            2
        };
        let mut input = encoding::Input::default();
        input.globals = vec![0.0; 32];
        input.globals[case] = 1.0;
        input.globals[4] = path.first().map_or(-1.0, |v| *v as f64);
        input.entities.tail.push([0.0; 64]);
        input.kinds.push(1);
        input.entity_mask.push(true);
        let mut choices = vec![];
        for index in 0..count {
            let mut row = [0.0; 64];
            row[index] = 1.0;
            row[48] = if path.is_empty() {
                1.0 / 16.0
            } else {
                4.0 / 16.0
            };
            input.candidates.push(row);
            input.sources.push(-1);
            input.targets.push(-1);
            input.candidate_mask.push(true);
            let mut command = Command::new("end");
            command.x = Some(index as f64);
            choices.push(McChoice {
                command,
                next: path.is_empty(),
                status: "available",
            });
        }
        Ok(McNode {
            stage: if path.is_empty() { "action" } else { "target" },
            input,
            choices,
        })
    };
    let (choice, path, metrics) = sampler::mc_search(
        &mut nodes,
        inference,
        &mut random,
        true,
        &mut |_, path, _, _| Ok(path != [2, 1]),
    )?;
    let pass = matches!(choice, Selected::Pass);
    let expectation = if pass {
        if case == 0 { 0.6 } else { -0.4 }
    } else if (case == 1 && path == [1, 1]) || (case == 2 && path == [2, 0]) {
        0.8
    } else {
        -0.8
    };
    let reward = if Random::new(outcome_seed).next() < (expectation + 1.0) / 2.0 {
        1
    } else {
        -1
    };
    Ok(
        json!({"type":"done","exercise":true,"case":case,"pass":pass,"path":path,"expectation":expectation,"reward":reward,"metrics":metrics}),
    )
}
