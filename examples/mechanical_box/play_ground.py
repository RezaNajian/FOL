import numpy as np
import orbax.checkpoint as ocp
import jax
from flax import nnx

temp_test_dir = '/home/reza/project/FOL_NNs_repo/feature_flax_optax_cfol/examples/mechanical_box/tmp'

if False:
    ckpt_dir = ocp.test_utils.erase_and_create_empty(temp_test_dir)

    class TwoLayerMLP(nnx.Module):
        def __init__(self, dim, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
            self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

        def __call__(self, x):
            x = self.linear1(x)
            return self.linear2(x)

    # Create this model and show we can run it
    model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
    # x = jax.random.normal(jax.random.key(42), (3, 4))
    # assert model(x).shape == (3, 4)


    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(temp_test_dir+"/check", state)
    checkpointer.wait_until_finished()

    state_restored = checkpointer.restore(temp_test_dir+"/check", state)
    jax.tree.map(np.testing.assert_array_equal, state, state_restored)
    print('NNX State restored successfully ! ')
    # nnx.display(state_restored)

    exit()



if True:
    path = ocp.test_utils.erase_and_create_empty(temp_test_dir)

    state = {
        'a': np.arange(8),
        'b': np.arange(16),
    }
    extra_params = [42, 43]

    class TwoLayerMLP(nnx.Module):
        def __init__(self, dim, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
            self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

        def __call__(self, x):
            x = self.linear1(x)
            return self.linear2(x)

    # Create this model and show we can run it
    model = TwoLayerMLP(4, rngs=nnx.Rngs(0))

    # Keeps a maximum of 3 checkpoints, and only saves every other step.
    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(
        path, options=options, item_names=('state', 'extra_params','model_state')
    )

    for step in range(11):  # [0, 1, ..., 10]
        _, model_state = nnx.split(model)
        mngr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                extra_params=ocp.args.JsonSave(extra_params),
                model_state = ocp.args.StandardSave(model_state)
            ),
        )
    mngr.wait_until_finished()
    restored = mngr.restore(10,args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            extra_params=ocp.args.JsonSave(extra_params),
            model_state = ocp.args.StandardSave(model_state)
        ))
    model_state, restored_extra_params = restored.model_state, restored.extra_params

    print(model_state)