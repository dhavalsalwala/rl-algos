from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.core.serializable import Serializable
import utils.common_utils as utils


class FirstOrderOptimizerExt(FirstOrderOptimizer, Serializable):

    def __init__(self, clip_grads=None, **kwargs):
        Serializable.quick_init(self, locals())
        self.clip_grads = clip_grads
        super(FirstOrderOptimizerExt, self).__init__(**kwargs)

    @overrides
    def update_opt(self, loss, target, inputs, extra_inputs=None):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param inputs: A list of symbolic variables as inputs
        :param extra_inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        grads = utils.clip_grads(
            self._tf_optimizer.compute_gradients(loss=loss, var_list=target.get_params(trainable=True)),
            self.clip_grads)
        self._train_op = self._tf_optimizer.apply_gradients(grads)

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
        )
