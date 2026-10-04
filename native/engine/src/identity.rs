//! 宿主可检查的构建身份；内部 Rust API 不在稳定协议承诺内。
use serde_json::{Value, json};

pub fn describe() -> Value {
    json!({
        "name": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "build": {
            "sourceSha256": env!("HAOJIE_SOURCE_SHA256"),
            "rustc": env!("HAOJIE_RUSTC"),
            "target": env!("HAOJIE_TARGET"),
            "profile": env!("HAOJIE_PROFILE"),
            "kernelProfile": cfg!(feature = "kernel-profile")
        },
        "ruleset": crate::model::RULESET,
        "rulesPackage": "haojie-rules-package-v1",
        "rulesHash": crate::records::rules_hash(),
        "encoding": "haojie-entities-factorized-v1",
        "record": crate::records::FORMAT,
        "auditor": crate::records::AUDITOR,
        "protocols": {"training": crate::training_host::PROTOCOL, "development": crate::model::PROTOCOL},
        "capabilities": ["create", "observe", "query", "preflight", "execute", "continuous-sampling", "public-encoding", "record-write", "record-audit", "audited-encoding", "learning-stage-resume", "memory-examples-v1", "mc-context-v2", "actor-model-routing-v1", "teacher-v2"],
        "interfaces": {
            "training": "validated-start-command-record-input",
            "development": "trusted-canonical-state-and-rule-package",
            "rust": "internal"
        },
        "resume": "python-checkpoint-at-complete-learning-stage"
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn identity_preserves_protocol_versions() {
        let identity = super::describe();
        assert_eq!(identity["name"], "haojie-engine");
        assert_eq!(identity["record"], "haojie-native-record-v1");
        assert_eq!(
            identity["build"]["sourceSha256"].as_str().unwrap().len(),
            64
        );
        assert_eq!(
            identity["build"]["kernelProfile"],
            cfg!(feature = "kernel-profile")
        );
    }
}
