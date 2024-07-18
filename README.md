
## Brain-Speech CLIP: Contrastive Learning for EEG and Speech Envelope Alignmen

In a situation where multiple conversations happening around us. Despite the noise, human listeners have an extraordinary ability to focus on a single speaker while ignoring others. Our project dives into this, hypothesizing that the speech representations we attend to align more closely with our EEG brainwave patterns than those we ignore.

We introduce **Brain-Speech CLIP**, a contrastive learning framework inspired by the CLIP model. Our innovative approach trains the model to maximize the similarity between EEG and speech envelope representations of attended speech, while minimizing the similarity for unattended speech.

### Key Features
- **Novel Contrastive Learning Framework**: Leverages the power of contrastive learning to bridge the gap between brainwave and speech representations.

- **Between experiment Generalization**: The model was trained on EEG data from one experiment and tested on EEG data from another experiment.

- **Between Task Generalization**: The model was trained on matching speech representations with EEG representations but tested on identifying the attended speech in a multi-talker scenario.

- **Zero-Shot Learning**: Capable of generalizing to new, unseen data without additional training.

### Results


