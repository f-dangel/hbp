"""Training procedure using a second-order method."""

import torch

from bpexts.hbp.loss import batch_summed_hessian

from .training import Training


class SecondOrderTraining(Training):
    """Handle logging in training procedure with 2nd-order optimizers."""

    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        data_loader,
        logdir,
        num_epochs,
        modify_2nd_order_terms,
        logs_per_epoch=10,
        device=None,
    ):
        """Train a model, log loss values into logdir.

        See `Training` class in `training.py`

        Parameters:
        -----------
        modify_2nd_order_terms : string ('none', 'clip', 'sign', 'zero')
                String specifying the strategy for dealing with 2nd-order
                module effects in Hessian backpropagation
        """
        if device is None:
            device = torch.device("cpu")

        super().__init__(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            num_epochs,
            logs_per_epoch=logs_per_epoch,
            device=device,
        )
        self.modify_2nd_order = modify_2nd_order_terms

    # override
    def backward_pass(self, outputs, loss):
        """Perform backward pass for gradients and Hessians.

        Parameters:
        -----------
        outputs : (torch.Tensor)
            Tensor of size (batch_size, ...) storing the model outputs
        loss : (torch.Tensor)
            Scalar value of the loss function evaluated on the outputs
        """
        # Hessian w.r.t the outputs
        output_hessian = self._compute_loss_hessian(outputs, loss)
        # compute gradients
        super().backward_pass(outputs, loss)
        # backward Hessian
        self.model.backward_hessian(
            output_hessian,
            compute_input_hessian=False,
            modify_2nd_order_terms=self.modify_2nd_order,
        )

    def _compute_loss_hessian(self, outputs, loss):
        """Hessian of the loss with respect to the outputs."""
        return batch_summed_hessian(loss, outputs)


class HBPSecondOrderTraining(SecondOrderTraining):
    # override
    def _compute_loss_hessian(self, outputs, loss):
        """Compute the batch-averaged Hessian."""
        return self.loss_function.batch_summed_hessian(loss, outputs)


class CVPSecondOrderTraining(SecondOrderTraining):
    # override
    def _compute_loss_hessian(self, outputs, loss):
        """Hessian of the loss with respect to the outputs.

        For torch losses, computes the batch-averaged Hessian.
        For CVP losses, computes exact Hessian-vector routines.
        """
        return self.loss_function.backward_hessian(None, compute_input_hessian=True)


class KFACTraining(Training):
    pass
