### Improving safety despite intentional subersion
Techniques that prevent AI catastrophes event if AI instances are colluding to subvert the safety techniques.
Ensure safety if the model itself is trying to subvert.
**I think that a more useful framework to think about this kind of stuff is to imagine a situation in which the roleplaying behavior of the model's gets out of hand and it's difficult to steer it back to "helpful, honest and harmless"**
Setup:
- Untrusted more capable model: GPT4
- Trusted less capable model: GPT3.5
- Task: Solving some programming questions without leaving backdors
- Backdoor: Solutions that passes all the tests but's incorrect in an input that GPT4 finds.
- The read team tries to find good stratefies to provide in the form of an attack policy.
**Trusted Monitoring**
**Blue Team protocol**
Code generation: ask U
Auditing decision: ask T whether the code looks suspicious, audit the top 2% most suspicious proposals
**Read Team strategy**
Write subtle backdooes, backdoors are only injected when there is a food opportunity to do so.
**Untrusted Monitoring**
Safety vs Usefulness tradeoffs
Usefulness is the correctness of the submitted solutions (0 = performance of GPT3.5 1= performance of GPT4)
Another effect of modeling scheming by explicitly red-teaming is that, because the read-team is currently much better at constructing strategies to evade safety measures that GPT4, we are able to empirically investigate more intricate attacks and defenses that scalable oversight projects do.
**Uppfront audditing**
Blue Team protocol: Produce code using U, Audit the produced code on the first 2% of problems.
Read Tram attack policy: On rare inputs, add an obvious backdoor to the code you write.
Upfront auditing is able to defeat some exteremly simple attack policies
This attack has to worry about the possibility that before it adds a backward to some code, the code is already wrong because of an honest mistake that will be caught in regular testing.
Backdooring incorrect code has no upside but downsides.
**LAT**
- Targeted: Pairs of wanted/unwanted responses 
- Untargeted: Steer Away from the desirable response
We attack the RS of transformer LLMs with L2-Norm-Bounded perturbations, calculated using projected gradient Descent
**Important**
Because the model and attacker are optimized using differnt completions to prompts, we only perturb the token positions in the residual stream corresponding to the prompt (and no to the completions), perturbations across layers perform better than single layer perturbations.




### Reflections on the feasibility of scalable oversight 


**MADGT** Make AI do great things


Scalable oversight is the method for training an AI using evaluations of what the AI is doing that can be scaled to superhuman intelligence.

Evaluations can take many forms, an evaluator might:
- Compare differnet proposed solutions
- Rate different solution
- Provide correction 
- Debate
- Market making




*Anthropic constitutional AI can be seen as a step towards scalable oversight with pure AI suppervision*


Aligning AI to a goal that is based on evaluations can only work to the extent that our evaluations fully capture, what we mean by "aligned"

**Failure Modes**

1) Unintended catastrophe:

If an AI where to be trained on the decisions of investment bankers pre-2008, it would have had a losing strategy.


2) The betrayal

If the evaluator is smarter than the evaluatee (because it is advantaged by the training process) it will to some degree be able to see through lies and misinformation. 




### Measuring progress in scalable oversight for LLMs


The goal of scalable oversight is difficult to pursue experimentally since current systems differ in many important ways from the future systems that we are most concerned with being able to align.


**Sandwitch experimental paradigm**

The model is more capable than the average human but not more capable than an expert.


**Proof of concept experiment**

We present human participants with difficult mulltiple choice questions from MMLU and QuALITY on which we expect our existing assistant to perform better than the participants could naïvely on their own.


### Debating with More Persuasive LLMs Leads to More Truthful Answers


**Can weaker models assess the correctness of stronger models?**

- Stronger models --> Experts (posses the info to answer questions)
- Weaker models --> Non Experts (lack the information but are as capable as the stronger model)



The method we evaluate is debate, where 2 LLMs experts each argue for a different answer,a nd non-experts selects the answer.


**Dataset:QuALITY** 


Debate consistently helps non-expert models and humans answer question, achieving 76% and 88% accuracy (48% and 60% baseline)


It's eassier to assess the correctness of an answer than to generate it, (Goodfellow et All)


To simulate stronger and weaker models we use information assymetri.


*Persuasiveness: Measured by judge approval (doesn't require ground truth)*


**Debate**

Two models argue for opposing answers to a question, after N rounds, a judge is tasked with picking the correct answer. 


**Interactive Debate**

Uses 2 experts arguing for opposing answers, in addition after each round the judge may make a statement.


**Consultancy (Baseline)**

- N rounds, between the agent and the judge.
- The agent argues for both the correct and incorrect responses in different rounds.


**Tasks**

- QuALITY (Gutemberg science fiction)
- We provide 2 answers 

**Quote verification tool**

Text search, depending on the result of the search the tool generates <v_quote> or <u_quote>, the judge is instructed to only trust the <v_quote>




























