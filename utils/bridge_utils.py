from models import BaseModel


def reset_weights(m):
    for layer in m.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


class AlphaScheduler:
    def __init__(self, milestones: list[int], alphas: list[float], alpha_start: float) -> None:
        self.counter = 0
        self.schedule = dict(zip(milestones, alphas))
        self.current_alpha = alpha_start

    @property
    def alpha(self) -> float:
        return self.current_alpha

    def step(self, bwd_model: BaseModel | None = None):
        self.counter += 1

        if self.counter in self.schedule:
            self.current_alpha = self.schedule[self.counter]
            if bwd_model is not None:
                bwd_model.apply(reset_weights)
