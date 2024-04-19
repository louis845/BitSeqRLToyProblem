# BitSeqRLToyProblem
Repo for bit sequence DQN RL

## Environment and packages
python==3.10.*

`pip install -r requirements.txt`

## Most updated documentation (with possible new research directions)
 * [most updated docs.pdf](https://github.com/louis845/BitSeqRLToyProblemPrivateDocs/blob/master/docs/docs.pdf)
 * We should keep this private for now, check your email!

## Documentation
 * [`docs/docs.pdf`](docs/docs.pdf)

## Important files

**Environment:**

 * [`src/environments/env_bit_sequence_flipping_rng_target.py`](./src/environments/env_bit_sequence_flipping_rng_target.py)

**Models**

 * [`src/models/model_dqn_bitflipping_target.py`](./src/models/model_dqn_bitflipping_target.py)

**Buffer**

 * [`src/models/buffer_base.py`](src/models/buffer_base.py)
 * [`src/models/buffer_bitflipping_target.py`](src/models/buffer_bitflipping_target.py)

**Agents**

 * [`src/agents/agent_dqn_target.py`](src/agents/agent_dqn_target.py)
 * [`src/agents/agent_dqn_target_her.py`](src/agents/agent_dqn_target_her.py)

**Training and evaluation script**

 * [`src/runtime.py`](src/runtime.py)

**Others**

 * [`src/utils.py`](src/utils.py)