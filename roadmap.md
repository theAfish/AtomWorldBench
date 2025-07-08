Organize result evaluation into three difficulty tiers, test structured manipulation actions, and benchmark across multiple LLM models.


## 1. Problem Categorization & Action Types

### Easy: simple atomic edits (no spatial reasoning)  
  - `AddAtomAction`  
  - `RemoveAtomAction`  
  - `ChangeAtomAction`  
Testing accurate data modification / fractional to Cartesian conversion.

### Medium: basic spatial operations  
  - `InsertBetweenAtomsAction`  
  - `DeleteAroundAtomAction`  
  - `MoveTowardsAtomAction`  
evaluating spatial imagination and localized structure manipulation.

### Hard: global structure understanding  
  - `Motif` related actions
Testing LLM’s understanding of overall 3D structure (waiting on Fengyu for action design).


## 2. Plan  
- For each task class, select 3 distinct Actions.  
- Generate 250 test instances per action.


## 3. Metrics  
- Use structureMatcher for output comparison:  
  1. max_dist for single‑atom edits  
  2. RMSD for multi‑atom / structure‑level edits  


## 4. Models Under Evaluation  
Aim to cover diverse LLM types and sizes:

`gpt-o3`, `gpt-o4`, `gpt-o4mini`, `llama3`, `gemini2.5`, `deepseek-chat`, `deepseek-reasoner`, `qwen3`


## To-Do  
- [ ] Fengyu: implement motif‑related actions and integration  
- [ ] Alex: run benchmark tests on ChatGPT, LLaMA, and Gemini models  
- [ ] Tau: run benchmark tests on DeepSeek‑chat, DeepSeek‑reasoner, and Qwen3  
- [ ] Other code related issues


