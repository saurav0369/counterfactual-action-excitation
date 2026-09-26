# DEV runtime implementation note

Before completing the frozen DEV grid, the first execution was terminated by the execution environment after only two factual-validation configurations had completed. No counterfactual evaluation was run or inspected.

A semantics-preserving runtime change was then made: PyTorch intra-op CPU threads were fixed to one (`torch.set_num_threads(1)`) to remove thread-launch overhead for the small MLPs. The candidate grid, data seeds, model seeds, training steps, batch size, optimizer, selection metric, and tie-break rules were unchanged. The entire DEV grid is rerun from the beginning after this change; partial outputs from the interrupted run are not used for selection.
