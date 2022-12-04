# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based multi-head attention layer."""


import collections
import math
import string

import numpy as np
import tensorflow as tf #.compat.v2 as tf
#from tensorflow.python import keras

from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
#from keras.layers import activation
#from keras.layers import regularization
#from tensorflow.python.keras.utils import tf_utils
#from keras.engine import base_layer
from tensorflow.python.keras.layers.core.einsum_dense import EinsumDense 
from tensorflow.python.keras import backend
import numbers

# isort: off
from tensorflow.python.platform import tf_logging as logging
#from tensorflow.python.util.tf_export import keras_export

_CHR_IDX = string.ascii_lowercase


def validate_axis(axis, input_shape):
    """Validate an axis value and returns its standardized form.
    Args:
      axis: Value to validate. Can be an integer or a list/tuple of integers.
        Integers may be negative.
      input_shape: Reference input shape that the axis/axes refer to.
    Returns:
      Normalized form of `axis`, i.e. a list with all-positive values.
    """
    input_shape = tf.TensorShape(input_shape)
    rank = len(input_shape)
    if not rank:
        raise ValueError(
            'Input has undefined rank. Received: input_shape={}'.format(input_shape))

    # Convert axis to list and resolve negatives
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)
    for idx, x in enumerate(axis):
        if x < 0:
            print(rank)
            print(x)
            axis[idx] = rank-1 #+ x

    # Validate axes
    for x in axis:
        if x < 0 or x >= rank:
            raise ValueError('Invalid value for `axis` argument. Expected 0 <= axis < inputs.rank (with inputs.rank={}). Received: axis={}'.format(rank, tuple(axis)))
    if len(axis) != len(set(axis)):
        raise ValueError('Duplicate axis: {}'.format(tuple(axis)))
    return axis


def is_tf_random_generator_enabled():
    """Check whether `tf.random.Generator` is used for RNG in Keras.
    Compared to existing TF stateful random ops, `tf.random.Generator` uses
    `tf.Variable` and stateless random ops to generate random numbers,
    which leads to better reproducibility in distributed training.
    Note enabling it might introduce some breakage to existing code,
    by producing differently-seeded random number sequences
    and breaking tests that rely on specific random numbers being generated.
    To disable the
    usage of `tf.random.Generator`, please use
    `tf.keras.backend.experimental.disable_random_generator`.
    We expect the `tf.random.Generator` code path to become the default, and
    will remove the legacy stateful random ops such as `tf.random.uniform` in
    the future (see the [TF RNG guide](
    https://www.tensorflow.org/guide/random_numbers)).
    This API will also be removed in a future release as well, together with
    `tf.keras.backend.experimental.enable_tf_random_generator()` and
    `tf.keras.backend.experimental.disable_tf_random_generator()`
    Returns:
      boolean: whether `tf.random.Generator` is used for random number
        generation in Keras.
    """
    return True


class RandomGenerator():
    """Random generator that selects appropriate random ops.
    This class contains the logic for legacy stateful random ops, as well as the
    new stateless random ops with seeds and tf.random.Generator. Any class that
    relies on RNG (eg initializer, shuffle, dropout) should use this class to
    handle the transition from legacy RNGs to new RNGs.
    Args:
      seed: Optional int seed. When `rng_type` is "stateful", the seed is used
        to create `tf.random.Generator` to produce deterministic sequences.
        When `rng_type` is "stateless", new seed will be created if it is not
        provided by user, and it will be passed down to stateless random ops.
        When `rng_type` is "legacy_stateful", the seed will be passed down to
        stateful random ops.
      rng_type: Type of RNG to use, one of "stateful", "stateless",
        "legacy_stateful". It defaults to "stateful" if
        `enable_tf_random_generator` has been activated, or to
        "legacy_stateful" otherwise.
        - When using "stateless", the random ops outputs are constant (the same
          inputs result in the same outputs).
        - When using "stateful" or "legacy_stateful", the random ops outputs are
          non-constant, but deterministic: calling the same random op multiple
          times with the same inputs results in a deterministic sequence of
          different outputs.
        - "legacy_stateful" is backed by TF1 stateful RNG ops
          (e.g. `tf.random.uniform`), while "stateful"
          is backed by TF2 APIs (e.g. `tf.random.Generator.uniform`).
    """

    RNG_STATELESS = "stateless"
    RNG_STATEFUL = "stateful"
    RNG_LEGACY_STATEFUL = "legacy_stateful"

    def __init__(self, seed=None, rng_type=None, **kwargs):
        self._seed = seed
        self._set_rng_type(rng_type, **kwargs)
        self._built = False

    def _set_rng_type(self, rng_type, **kwargs):
        # Only supported kwargs is "force_generator", which we will remove once
        # we clean up all the caller.
        # TODO(scottzhu): Remove the kwargs for force_generator.
        if kwargs.get("force_generator", False):
            rng_type = self.RNG_STATEFUL
        if rng_type is None:
            if is_tf_random_generator_enabled():
                self._rng_type = self.RNG_STATEFUL
            else:
                self._rng_type = self.RNG_LEGACY_STATEFUL
        else:
            if rng_type not in [
                self.RNG_STATEFUL,
                self.RNG_LEGACY_STATEFUL,
                self.RNG_STATELESS,
            ]:
                raise ValueError("error")
            self._rng_type = rng_type

    def _maybe_init(self):
        """Lazily init the RandomGenerator.
        The TF API executing_eagerly_outside_functions() has some side effect,
        and couldn't be used before API like tf.enable_eager_execution(). Some
        of the client side code was creating the initializer at the code load
        time, which triggers the creation of RandomGenerator. Lazy init this
        class to walkaround this issue until it is resolved on TF side.
        """
        # TODO(b/167482354): Change this back to normal init when the bug is
        # fixed.
        if self._built:
            return

        if self._rng_type == self.RNG_STATEFUL:
            # Fall back to legacy stateful since the generator need to work in
            # tf2.
            self._rng_type = self.RNG_LEGACY_STATEFUL

        if self._rng_type == self.RNG_STATELESS:
            self._seed = self._create_seed(self._seed)
            self._generator = None
        elif self._rng_type == self.RNG_STATEFUL:
            with tf_utils.maybe_init_scope(self):
                seed = self._create_seed(self._seed)
                self._generator = tf.random.Generator.from_seed(
                    seed, alg=tf.random.Algorithm.AUTO_SELECT
                )
        else:
            # In legacy stateful, we use stateful op, regardless whether user
            # provide seed or not. Seeded stateful op will ensure generating
            # same sequences.
            self._generator = None
        self._built = True

    def make_seed_for_stateless_op(self):
        """Generate a new seed based on the init config.
        Note that this will not return python ints which will be frozen in the
        graph and cause stateless op to return the same value. It will only
        return value when generator is used, otherwise it will return None.
        Returns:
          A tensor with shape [2,].
        """
        self._maybe_init()
        if self._rng_type == self.RNG_STATELESS:
            return [self._seed, 0]
        elif self._rng_type == self.RNG_STATEFUL:
            return self._generator.make_seeds()[:, 0]
        return None

    def make_legacy_seed(self):
        """Create a new seed for the legacy stateful ops to use.
        When user didn't provide any original seed, this method will return
        None.  Otherwise it will increment the counter and return as the new
        seed.
        Note that it is important to generate different seed for stateful ops in
        the `tf.function`. The random ops will return same value when same seed
        is provided in the `tf.function`.
        Returns:
          int as new seed, or None.
        """
        if self._seed is not None:
            result = self._seed
            self._seed += 1
            return result
        return None

    def _create_seed(self, user_specified_seed):
        if user_specified_seed is not None:
            return user_specified_seed
        elif getattr(_SEED_GENERATOR, "generator", None):
            return _SEED_GENERATOR.generator.randint(1, 1e9)
        else:
            return random.randint(1, int(1e9))

    def random_normal(
        self, shape, mean=0.0, stddev=1.0, dtype=None, nonce=None
    ):
        """Produce random number based on the normal distribution.
        Args:
          shape: The shape of the random values to generate.
          mean: Floats, default to 0. Mean of the random values to generate.
          stddev: Floats, default to 1. Standard deviation of the random values
            to generate.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.normal(
                shape=shape, mean=mean, stddev=stddev, dtype=dtype
            )
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)
            return tf.random.stateless_normal(
                shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
            )
        return tf.random.normal(
            shape=shape,
            mean=mean,
            stddev=stddev,
            dtype=dtype,
            seed=self.make_legacy_seed(),
        )

    def random_uniform(
        self, shape, minval=0.0, maxval=None, dtype=None, nonce=None
    ):
        """Produce random number based on the uniform distribution.
        Args:
          shape: The shape of the random values to generate.
          minval: Floats, default to 0. Lower bound of the range of
            random values to generate (inclusive).
          minval: Floats, default to None. Upper bound of the range of
            random values to generate (exclusive).
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.uniform(
                shape=shape, minval=minval, maxval=maxval, dtype=dtype
            )
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)
            return tf.random.stateless_uniform(
                shape=shape,
                minval=minval,
                maxval=maxval,
                dtype=dtype,
                seed=seed,
            )
        return tf.random.uniform(
            shape=shape,
            minval=minval,
            maxval=maxval,
            dtype=dtype,
            seed=self.make_legacy_seed(),
        )

    def truncated_normal(
        self, shape, mean=0.0, stddev=1.0, dtype=None, nonce=None
    ):
        """Produce random number based on the truncated normal distribution.
        Args:
          shape: The shape of the random values to generate.
          mean: Floats, default to 0. Mean of the random values to generate.
          stddev: Floats, default to 1. Standard deviation of the random values
            to generate.
          dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise (via
            `tf.keras.backend.set_floatx(float_dtype)`)
          nonce: Optional integer scalar, that will be folded into the seed in
            the stateless mode.
        """
        self._maybe_init()
        dtype = dtype or floatx()
        if self._rng_type == self.RNG_STATEFUL:
            return self._generator.truncated_normal(
                shape=shape, mean=mean, stddev=stddev, dtype=dtype
            )
        elif self._rng_type == self.RNG_STATELESS:
            seed = self.make_seed_for_stateless_op()
            if nonce:
                seed = tf.random.experimental.stateless_fold_in(seed, nonce)
            return tf.random.stateless_truncated_normal(
                shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
            )
        return tf.random.truncated_normal(
            shape=shape,
            mean=mean,
            stddev=stddev,
            dtype=dtype,
            seed=self.make_legacy_seed(),
        )

    def dropout(self, inputs, rate, noise_shape=None):
        self._maybe_init()
        if self._rng_type in [self.RNG_STATEFUL, self.RNG_STATELESS]:
            return tf.nn.experimental.stateless_dropout(
                inputs,
                rate=rate,
                noise_shape=noise_shape,
                seed=self.make_seed_for_stateless_op(),
            )
        return tf.nn.dropout(
            inputs,
            rate=rate,
            noise_shape=noise_shape,
            seed=self.make_legacy_seed(),
        )






class LayerNormalization(Layer):
    """Layer normalization layer (Ba et al., 2016).
    Normalize the activations of the previous layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within each
    example close to 0 and the activation standard deviation close to 1.
    Given a tensor `inputs`, moments are calculated and normalization
    is performed across the axes specified in `axis`.
    Example:
    >>> data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
    >>> print(data)
    tf.Tensor(
    [[ 0. 10.]
     [20. 30.]
     [40. 50.]
     [60. 70.]
     [80. 90.]], shape=(5, 2), dtype=float32)
    >>> layer = tf.keras.layers.LayerNormalization(axis=1)
    >>> output = layer(data)
    >>> print(output)
    tf.Tensor(
    [[-1. 1.]
     [-1. 1.]
     [-1. 1.]
     [-1. 1.]
     [-1. 1.]], shape=(5, 2), dtype=float32)
    Notice that with Layer Normalization the normalization happens across the
    axes *within* each example, rather than across different examples in the
    batch.
    If `scale` or `center` are enabled, the layer will scale the normalized
    outputs by broadcasting them with a trainable variable `gamma`, and center
    the outputs by broadcasting with a trainable variable `beta`. `gamma` will
    default to a ones tensor and `beta` will default to a zeros tensor, so that
    centering and scaling are no-ops before training has begun.
    So, with scaling and centering enabled the normalization equations
    are as follows:
    Let the intermediate activations for a mini-batch to be the `inputs`.
    For each sample `x_i` in `inputs` with `k` features, we compute the mean and
    variance of the sample:
    ```python
    mean_i = sum(x_i[j] for j in range(k)) / k
    var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
    ```
    and then compute a normalized `x_i_normalized`, including a small factor
    `epsilon` for numerical stability.
    ```python
    x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
    ```
    And finally `x_i_normalized ` is linearly transformed by `gamma` and `beta`,
    which are learned parameters:
    ```python
    output_i = x_i_normalized * gamma + beta
    ```
    `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
    this part of the inputs' shape must be fully defined.
    For example:
    >>> layer = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])
    >>> layer.build([5, 20, 30, 40])
    >>> print(layer.beta.shape)
    (20, 30, 40)
    >>> print(layer.gamma.shape)
    (20, 30, 40)
    Note that other implementations of layer normalization may choose to define
    `gamma` and `beta` over a separate set of axes from the axes being
    normalized across. For example, Group Normalization
    ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
    corresponds to a Layer Normalization that normalizes across height, width,
    and channel and has `gamma` and `beta` span only the channel dimension.
    So, this Layer Normalization implementation will not match a Group
    Normalization layer with group size set to 1.
    Args:
      axis: Integer or List/Tuple. The axis or axes to normalize across.
        Typically this is the features axis/axes. The left-out axes are
        typically the batch axis/axes. This argument defaults to `-1`, the last
        dimension in the input.
      epsilon: Small float added to variance to avoid dividing by zero. Defaults
        to 1e-3
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored. Defaults to True.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling will be done by the next layer.
      beta_initializer: Initializer for the beta weight. Defaults to zeros.
      gamma_initializer: Initializer for the gamma weight. Defaults to ones.
      beta_regularizer: Optional regularizer for the beta weight. None by
        default.
      gamma_regularizer: Optional regularizer for the gamma weight. None by
        default.
      beta_constraint: Optional constraint for the beta weight. None by default.
      gamma_constraint: Optional constraint for the gamma weight. None by
        default.
    Input shape:
      Arbitrary. Use the keyword argument `input_shape` (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.
    Output shape:
      Same shape as input.
    Reference:
      - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
    """

    #@utils.allow_initializer_layout
    def __init__(
        self,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis)
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.supports_masking = True

        # Indicates whether a faster fused implementation can be used. This will
        # be set to True or False in build()"
        self._fused = None

    def _fused_can_be_used(self, ndims):
        """Returns false if fused implementation cannot be used.
        Check if the axis is contiguous and can be collapsed into the last axis.
        The self.axis is assumed to have no duplicates.
        """
        axis = sorted(self.axis)
        can_use_fused = False

        if axis[-1] == ndims - 1 and axis[-1] - axis[0] == len(axis) - 1:
            can_use_fused = True

        # fused_batch_norm will silently raise epsilon to be at least 1.001e-5,
        # so we cannot used the fused version if epsilon is below that value.
        # Also, the variable dtype must be float32, as fused_batch_norm only
        # supports float32 variables.
        if self.epsilon < 1.001e-5 or self.dtype != "float32":
            can_use_fused = False

        return can_use_fused

    def build(self, input_shape):
        self.axis = validate_axis(self.axis, input_shape)
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.dims

        param_shape = [input_shape[dim] for dim in self.axis]
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                #experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                #experimental_autocast=False,
            )
        else:
            self.beta = None

        self._fused = False #self._fused_can_be_used(rank)
        self.built = True

    def call(self, inputs):
        # TODO(b/229545225): Remove the RaggedTensor check.
        #is_ragged = False #isinstance(inputs, tf.RaggedTensor)
        #if is_ragged:
        #    inputs_lengths = inputs.nested_row_lengths()
        #    inputs = inputs.to_tensor()
        #inputs = tf.cast(inputs, )
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)

        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape.dims[dim].value

        def _broadcast(v):
            if (
                v is not None
                and len(v.shape) != ndims
                and self.axis != [ndims - 1]
            ):
                return tf.reshape(v, broadcast_shape)
            return v

        if not self._fused:
            input_dtype = inputs.dtype
            if (
                input_dtype in ("float16", "bfloat16")
                and self.dtype == "float32"
            ):
                # If mixed precision is used, cast inputs to float32 so that
                # this is at least as numerically stable as the fused version.
                inputs = tf.cast(inputs, "float32")

            # Calculate the moments on the last axis (layer activations).
            mean, variance = tf.nn.moments(inputs, self.axis) #, keepdims=True)

            scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

            # Compute layer normalization using the batch_normalization
            # function.
            outputs = tf.nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset=offset,
                scale=scale,
                variance_epsilon=self.epsilon,
            )
            #outputs = tf.cast(outputs, input_dtype)
        else:
            # Collapse dims before self.axis, and dims in self.axis
            pre_dim, in_dim = (1, 1)
            axis = sorted(self.axis)
            tensor_shape = tf.shape(inputs)
            for dim in range(0, ndims):
                dim_tensor = tensor_shape[dim]
                if dim < axis[0]:
                    pre_dim = pre_dim * dim_tensor
                else:
                    assert dim in axis
                    in_dim = in_dim * dim_tensor

            squeezed_shape = [1, pre_dim, in_dim, 1]
            # This fused operation requires reshaped inputs to be NCHW.
            data_format = "NCHW"

            inputs = tf.reshape(inputs, squeezed_shape)

            # self.gamma and self.beta have the wrong shape for
            # fused_batch_norm, so we cannot pass them as the scale and offset
            # parameters. Therefore, we create two constant tensors in correct
            # shapes for fused_batch_norm and later construct a separate
            # calculation on the scale and offset.
            scale = tf.ones([pre_dim], dtype=self.dtype)
            offset = tf.zeros([pre_dim], dtype=self.dtype)

            # Compute layer normalization using the fused_batch_norm function.
            outputs, _, _ = tf.compat.v1.nn.fused_batch_norm(
                inputs,
                scale=scale,
                offset=offset,
                epsilon=self.epsilon,
                data_format=data_format,
            )

            outputs = tf.reshape(outputs, tensor_shape)

            scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

            if scale is not None:
                outputs = outputs * tf.cast(scale, outputs.dtype)
            if offset is not None:
                outputs = outputs + tf.cast(offset, outputs.dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        #if is_ragged:
        #    outputs = tf.RaggedTensor.from_tensor(outputs, inputs_lengths)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))






class BaseRandomLayer(Layer):
    """A layer handle the random number creation and savemodel behavior."""

    #@tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(
        self, seed=None, force_generator=False, rng_type=None
    ):
        """Initialize the BaseRandomLayer.
        Note that the constructor is annotated with
        @no_automatic_dependency_tracking. This is to skip the auto
        tracking of self._random_generator instance, which is an AutoTrackable.
        The backend.RandomGenerator could contain a tf.random.Generator instance
        which will have tf.Variable as the internal state. We want to avoid
        saving that state into model.weights and checkpoints for backward
        compatibility reason. In the meantime, we still need to make them
        visible to SavedModel when it is tracing the tf.function for the
        `call()`.
        See _list_extra_dependencies_for_serialization below for more details.
        Args:
          seed: optional integer, used to create RandomGenerator.
          force_generator: boolean, default to False, whether to force the
            RandomGenerator to use the code branch of tf.random.Generator.
          rng_type: string, the rng type that will be passed to backend
            RandomGenerator. Default to `None`, which will allow RandomGenerator
            to choose types by itself. Valid values are "stateful", "stateless",
            "legacy_stateful".
          **kwargs: other keyword arguments that will be passed to the parent
            *class
        """
        super(BaseRandomLayer, self).__init__()
        self._random_generator = RandomGenerator(
            seed, force_generator=force_generator, rng_type=rng_type
        )

    def build(self, input_shape):
        super(BaseRandomLayer, self).build(input_shape)
        self._random_generator._maybe_init()

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        if save_type == "savedmodel":
            cache = kwargs["cache"]
            # TODO(b/213628533): This must be called before super() to ensure
            # that any input shape changes are applied before getting the config
            # of the model.
            children = self._trackable_saved_model_saver.trackable_children(
                cache
            )
            # This method exposes the self._random_generator to SavedModel only
            # (not layer.weights and checkpoint).
            children["_random_generator"] = self._random_generator
        else:
            children = {}
        children.update(super()._trackable_children(save_type, **kwargs))
        return children

    def _lookup_dependency(self, name):
        # When loading from a Keras SavedModel load, make sure that the loader
        # can find the random generator, otherwise the loader will assume that
        # it does not exist, and will try to create a new generator.
        if name == "_random_generator":
            return self._random_generator
        else:
            return super()._lookup_dependency(name)




#@keras_export("keras.layers.Dropout")
class KerasDropout(BaseRandomLayer):
    """Applies Dropout to the input.
    The Dropout layer randomly sets input units to 0 with a frequency of `rate`
    at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
    all inputs is unchanged.
    Note that the Dropout layer only applies when `training` is set to True
    such that no values are dropped during inference. When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.
    (This is in contrast to setting `trainable=False` for a Dropout layer.
    `trainable` does not affect the layer's behavior, as Dropout does
    not have any variables/weights that can be frozen during training.)
    >>> tf.random.set_seed(0)
    >>> layer = tf.keras.layers.Dropout(.2, input_shape=(2,))
    >>> data = np.arange(10).reshape(5, 2).astype(np.float32)
    >>> print(data)
    [[0. 1.]
     [2. 3.]
     [4. 5.]
     [6. 7.]
     [8. 9.]]
    >>> outputs = layer(data, training=True)
    >>> print(outputs)
    tf.Tensor(
    [[ 0.    1.25]
     [ 2.5   3.75]
     [ 5.    6.25]
     [ 7.5   8.75]
     [10.    0.  ]], shape=(5, 2), dtype=float32)
    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the
        binary dropout mask that will be multiplied with the input.
        For instance, if your inputs have shape
        `(batch_size, timesteps, features)` and
        you want the dropout mask to be the same for all timesteps,
        you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.
    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(KerasDropout, self).__init__(seed=seed, **kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError("Invalid value {} received for `rate`, expected a value between 0 and 1.".format(rate))
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        # Subclasses of `Dropout` may implement `_get_noise_shape(self,
        # inputs)`, which will override `self.noise_shape`, and allows for
        # custom noise shapes with dynamically sized inputs.
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = tf.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(
                concrete_inputs_shape[i] if value is None else value
            )
        return tf.convert_to_tensor(noise_shape)

    def call(self, inputs, training=None):
        if isinstance(self.rate, numbers.Real) and self.rate == 0:
            return tf.identity(inputs)

        if training is None:
            training = backend.learning_phase()

        def dropped_inputs():
            return self._random_generator.dropout(
                inputs, self.rate, noise_shape=self._get_noise_shape(inputs)
            )

        output = control_flow_util.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "rate": self.rate,
            "noise_shape": self.noise_shape,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))




#@keras_export("keras.layers.Softmax")
class KerasSoftmax(Layer):
    """Softmax activation function.
    Example without mask:
    >>> inp = np.asarray([1., 2., 1.])
    >>> layer = tf.keras.layers.Softmax()
    >>> layer(inp).numpy()
    array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
    >>> mask = np.asarray([True, False, True], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([0.5, 0. , 0.5], dtype=float32)
    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
    Output shape:
      Same shape as the input.
    Args:
      axis: Integer, or list of Integers, axis along which the softmax
        normalization is applied.
    Call arguments:
      inputs: The inputs, or logits to the softmax layer.
      mask: A boolean mask of the same shape as `inputs`. Defaults to `None`.
        The mask specifies 1 to keep and 0 to mask.
    Returns:
      softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super(KerasSoftmax, self).__init__()
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype)
            )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(
                    inputs
                    - tf.reduce_logsumexp(inputs, axis=self.axis, keepdims=True)
                )
            else:
                return backend.softmax(inputs, axis=self.axis[0])
        return backend.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    #@tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape





def _build_attention_equation(rank, attn_axes):
    """Builds einsum equations for the attention computation.

    Query, key, value inputs after projection are expected to have the shape as:
    `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
    `bs` and `<non-attention dims>` are treated as `<batch dims>`.

    The attention operations can be generalized:
    (1) Query-key dot product:
    `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
    <key attention dims>, num_heads, channels) -> (<batch dims>,
    num_heads, <query attention dims>, <key attention dims>)`
    (2) Combination:
    `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
    (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch
    dims>, <query attention dims>, num_heads, channels)`

    Args:
      rank: Rank of query, key, value tensors.
      attn_axes: List/tuple of axes, `[-1, rank)`,
        that attention will be applied to.

    Returns:
      Einsum equations.
    """
    target_notation = _CHR_IDX[:rank]
    # `batch_dims` includes the head dim.
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join(
        [target_notation[i] for i in batch_dims]
        + [target_notation[i] for i in attn_axes]
        + [source_notation[i] for i in attn_axes]
    )
    dot_product_equation = "%s,%s->%s" % (
        source_notation,
        target_notation,
        product_notation,
    )
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (
        product_notation,
        source_notation,
        target_notation,
    )
    return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "{},{}->{}".format(input_str,kernel_str,output_str)

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


#@keras_export("keras.layers.MultiHeadAttention")
class MultiHeadAttention(Layer):
    """MultiHeadAttention layer.

    This is an implementation of multi-headed attention as described in the
    paper "Attention is all you Need" (Vaswani et al., 2017).
    If `query`, `key,` `value` are the same, then
    this is self-attention. Each timestep in `query` attends to the
    corresponding sequence in `key`, and returns a fixed-width vector.

    This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, key_dim)`,
    `(batch_size, <key/value dimensions>, value_dim)`.

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor.

    Finally, the result tensor with the last dimension as value_dim can take an
    linear projection and return.

    When using `MultiHeadAttention` inside a custom layer, the custom layer must
    implement its own `build()` method and call `MultiHeadAttention`'s
    `_build_from_signature()` there.
    This enables weights to be restored correctly when the model is loaded.

    Examples:

    Performs 1D cross-attention over two sequence inputs with an attention mask.
    Returns the additional attention weights over heads.

    >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
    >>> target = tf.keras.Input(shape=[8, 16])
    >>> source = tf.keras.Input(shape=[4, 16])
    >>> output_tensor, weights = layer(target, source,
    ...                                return_attention_scores=True)
    >>> print(output_tensor.shape)
    (None, 8, 16)
    >>> print(weights.shape)
    (None, 2, 8, 4)

    Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

    >>> layer = MultiHeadAttention(
    ...     num_heads=2, key_dim=2, attention_axes=(2, 3))
    >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
    >>> output_tensor = layer(input_tensor, input_tensor)
    >>> print(output_tensor.shape)
    (None, 5, 3, 4, 16)

    Args:
      num_heads: Number of attention heads.
      key_dim: Size of each attention head for query and key.
      value_dim: Size of each attention head for value.
      dropout: Dropout probability.
      use_bias: Boolean, whether the dense layers use bias vectors/matrices.
      output_shape: The expected shape of an output tensor, besides the batch
        and sequence dims. If not specified, projects back to the key feature
        dim.
      attention_axes: axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.

    Call arguments:
      query: Query `Tensor` of shape `(B, T, dim)`.
      value: Value `Tensor` of shape `(B, S, dim)`.
      key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. The boolean mask specifies which query
        elements can attend to which key elements, 1 indicates attention and 0
        indicates no attention. Broadcasting can happen for the missing batch
        dimensions and the head dimension.
      return_attention_scores: A boolean to indicate whether the output should
        be `(attention_output, attention_scores)` if `True`, or
        `attention_output` if `False`. Defaults to `False`.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
        Defaults to either using the training mode of the parent layer/model,
        or False (inference) if there is no parent layer.
      use_causal_mask: A boolean to indicate whether to apply a causal mask to
        prevent tokens from attending to future tokens (e.g., used in a decoder
        Transformer).

    Returns:
      attention_output: The result of the computation, of shape `(B, T, E)`,
        where `T` is for target sequence shapes and `E` is the query input last
        dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
        are projected to the shape specified by `output_shape`.
      attention_scores: [Optional] multi-head attention coefficients over
        attention axes.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        #**kwargs,
    ):
        super(MultiHeadAttention, self).__init__()
        self.supports_masking = True
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim if value_dim else key_dim
        self._dropout = dropout
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        if attention_axes is not None:
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False
        self._query_shape, self._key_shape, self._value_shape = None, None, None

    def get_config(self):
        config = {
            "num_heads": self._num_heads,
            "key_dim": self._key_dim,
            "value_dim": self._value_dim,
            "dropout": self._dropout,
            "use_bias": self._use_bias,
            "output_shape": self._output_shape,
            "attention_axes": self._attention_axes,
            "kernel_initializer": initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self._bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self._bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self._kernel_constraint),
            "bias_constraint": constraints.serialize(self._bias_constraint),
            "query_shape": self._query_shape,
            "key_shape": self._key_shape,
            "value_shape": self._value_shape,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        # If the layer has a different build() function from the Keras default,
        # we need to trigger the customized build to create weights.
        query_shape = config.pop("query_shape")
        key_shape = config.pop("key_shape")
        value_shape = config.pop("value_shape")
        layer = cls(**config)
        if None in [query_shape, key_shape, value_shape]:
            logging.warning(
                "One of dimensions of the input shape is missing. It "
                "should have been memorized when the layer was serialized. "
                "%s is created without weights.",
                str(cls),
            )
        else:
            layer._build_from_signature(query_shape, value_shape, key_shape)
        return layer

    def _build_from_signature(self, query, value, key=None):
        """Builds layers and variables.

        Once the method is called, self._built_from_signature will be set to
        True.

        Args:
          query: Query tensor or TensorShape.
          value: Value tensor or TensorShape.
          key: Key tensor or TensorShape.
        """
        self._built_from_signature = True
        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)
        if hasattr(value, "shape"):
            self._value_shape = tf.TensorShape(value.shape)
        else:
            self._value_shape = tf.TensorShape(value)
        if key is None:
            self._key_shape = self._value_shape
        elif hasattr(key, "shape"):
            self._key_shape = tf.TensorShape(key.shape)
        else:
            self._key_shape = tf.TensorShape(key)

        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        #with tf_utils.maybe_init_scope(self):
        #print(self._query_shape)
        #print(self._query_shape.dims)
        free_dims = len(self._query_shape.dims) - 1
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=1, output_dims=2
        )
        self._query_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            #self._get_common_kwargs_for_sublayer(),
        )
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            len(self._key_shape.dims) - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            #self._get_common_kwargs_for_sublayer(),
        )
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            len(self._value_shape.dims) - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            #self._get_common_kwargs_for_sublayer(),
        )

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports mult-head einsum computations.
        self._build_attention(output_rank)
        self._output_dense = self._make_output_dense(
            free_dims,
            self._get_common_kwargs_for_sublayer(),
            "attention_output",
        )

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def _make_output_dense(self, free_dims, common_kwargs, name=None):
        """Builds the output projection matrix.

        Args:
          free_dims: Number of free dimensions for einsum equation building.
          common_kwargs: Common keyword arguments for einsum layer.
          name: Name for the projection layer.

        Returns:
          Projection layer.
        """
        if self._output_shape:
            if not isinstance(self._output_shape, collections.abc.Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [self._query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=2, output_dims=len(output_shape)
        )
        return EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            #common_kwargs
        )

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
          rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(
            range(
                attn_scores_rank - len(self._attention_axes), attn_scores_rank
            )
        )
        self._softmax = KerasSoftmax(axis=norm_axes)
        self._dropout_layer = KerasDropout(rate=self._dropout)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. It is generally not needed if the
            `query` and `value` (and/or `key`) are masked.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        #attention_scores = self._masked_softmax(
        #    attention_scores, attention_mask
        #)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

    def call(self,query,value=None, key=None, attention_mask=None,return_attention_scores=False,training=None,use_causal_mask=False):
        
        if value==None:
            value = query #self attention
        
        attention_mask = self._compute_attention_mask(
            query,
            value,
            key=key,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value


        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def _compute_attention_mask(
        self, query, value, key=None, attention_mask=None, use_causal_mask=False
    ):
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions.
          use_causal_mask: A boolean to indicate whether to apply a causal mask
            to prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

        Returns:
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions, based on the Keras masks of the
            `query`, `key`, `value`, and `attention_mask` tensors, and the
            causal mask if `use_causal_mask=True`.
        """
        query_mask = getattr(query, "_keras_mask", None)
        value_mask = getattr(value, "_keras_mask", None)
        key_mask = getattr(key, "_keras_mask", None)
        auto_mask = None
        if query_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
            # B = batch size, T = max query length
            auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
            # B = batch size, S == max value length
            mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else tf.cast(attention_mask, bool) & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean `Tensor` equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```
        Args:
          query: query `Tensor` of shape `(B, T, ...)`.
          value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
          query).

        Returns:
          mask: a boolean `Tensor` of shape [1, T, S] containing a lower
                triangular matrix of shape [T, S].
        """
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )

    def compute_output_shape(self, query_shape, value_shape, key_shape=None):

        if key_shape is None:
            key_shape = value_shape

        query_shape = tf.TensorShape(query_shape)
        value_shape = tf.TensorShape(value_shape)
        key_shape = tf.TensorShape(key_shape)

        if query_shape[-1] != value_shape[-1]:
            raise ValueError("The last dimension of `query_shape` and `value_shape` must be equal, but are {}, {}. Received: query_shape={}, value_shape={}".format(query_shape[-1], value_shape[-1], query_shape, value_shape) )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError("All dimensions of `value` and `key` except the last one must be equal. Received {} and {}".format(value_shape, key_shape))

        if self._output_shape:
            return query_shape[:-1].concatenate(self._output_shape)

        return query_shape
