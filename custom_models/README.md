# Customer-trained machine types

`python train_custom.py ...` writes one folder per customer model here:
`model.pkl`, `meta.pkl`, `type.json` (what the app shows) and `metrics.json`
(how it validated). The backend registers every folder at startup, so after
training, restart the backend (or rebuild the Docker image) and the new type
appears in the machine-type dropdown.

Customer models are trained on that customer's data -- keep their folders out
of any public repository unless the customer agrees in writing.
