# Protocol patch v0.2.1

Before confirmatory execution and before any confirmatory output existed, two clarifications were frozen:

1. Primary factual and counterfactual errors are MSE to the known noiseless conditional transition mean, rather than NRMSE against noisy sampled outcomes. This isolates transition-estimation error from irreducible observation noise and avoids normalization differences induced by the two behavior distributions.
2. Factual and counterfactual evaluation sets use 16,384 examples, which is divisible by every denominator in the registered epsilon sweep through 1/256 and permits exactly balanced support constructions.

No historical exploratory result is used as confirmatory evidence.
