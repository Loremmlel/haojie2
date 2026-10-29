#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}
pub struct Settings {
    pub width: usize,
    pub depth: usize,
    pub per_action: usize,
    pub partial: usize,
    pub samples: usize,
    pub chance_limit: usize,
    pub nodes: usize,
    pub turn_nodes: usize,
    pub reply_roots: usize,
    pub noise: f64,
}
impl Difficulty {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "easy" => Ok(Self::Easy),
            "medium" => Ok(Self::Medium),
            "hard" => Ok(Self::Hard),
            _ => Err("unknown AI difficulty".into()),
        }
    }
    pub fn settings(self) -> Settings {
        match self {
            Self::Easy => Settings {
                width: 1,
                depth: 1,
                per_action: 3,
                partial: 4,
                samples: 2,
                chance_limit: 8,
                nodes: 100,
                turn_nodes: 1200,
                reply_roots: 0,
                noise: 10.,
            },
            Self::Medium => Settings {
                width: 2,
                depth: 5,
                per_action: 5,
                partial: 6,
                samples: 3,
                chance_limit: 12,
                nodes: 320,
                turn_nodes: 3200,
                reply_roots: 0,
                noise: 0.,
            },
            Self::Hard => Settings {
                width: 2,
                depth: 7,
                per_action: 4,
                partial: 5,
                samples: 4,
                chance_limit: 18,
                nodes: 800,
                turn_nodes: 8000,
                reply_roots: 2,
                noise: 0.,
            },
        }
    }
}
