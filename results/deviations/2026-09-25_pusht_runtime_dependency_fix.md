# PushT runtime dependency fix

The frozen external protocol specified `gym-pusht==0.1.6` but the initial Colab install resolved to a Pymunk 7.x runtime. `gym-pusht==0.1.6` calls `Space.add_collision_handler`, which is unavailable in Pymunk 7.x, so the external runner failed during the first environment reset before any PushT transition result or model metric was produced.

The runtime was corrected by installing `pymunk==6.10.0`, after which the unchanged frozen runner completed all 12 replicates. No scientific parameter changed: anchor rules, action scale, epsilon values, seeds, model configuration, thresholds, and analysis code remained unchanged.

This is a dependency-compatibility correction, not a scientific protocol change. The original frozen `requirements-pusht.txt` is preserved unchanged for hash integrity. `requirements-pusht-runtime.txt` records the runtime combination that successfully executed the experiment.
