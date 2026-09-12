"""Preserve one call receipt while replacing the outer inference model."""
from forward_ledger import ForwardLedger


class RetargetableForwardLedger(ForwardLedger):
    def retarget(self, model):
        assert self.handles and self.active is None and self.pending is None
        assert model is not self.model
        # No model call may occur between these synchronous operations.
        super().__exit__(None,None,None)
        self.model=model
        super().__enter__()
