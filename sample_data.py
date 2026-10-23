"""Sample datasets for SmartTopic_LDA topic modeling demonstration."""

import pandas as pd

TECH_AI_DOCUMENTS = [
    # Topic: Deep Learning & Neural Networks
    "Transformers and attention mechanisms have revolutionized natural language processing and computer vision applications.",
    "Convolutional neural networks extract spatial features from images, enabling accurate object detection and segmentation.",
    "Deep reinforcement learning agents learn optimal control policies through trial, reward signals, and policy gradients.",
    "Backpropagation computes gradient vectors through computational graphs using the chain rule for parameter optimization.",
    "Diffusion models generate photorealistic synthetic imagery by progressively reversing stochastic Gaussian noise processes.",
    "Large language models trained on massive corpora demonstrate few-shot reasoning, code generation, and chain-of-thought capabilities.",
    "Gradient clipping and weight decay prevent exploding gradients and overfitting during deep neural network training iterations.",
    "Vision transformers split images into patch embeddings and process visual tokens with multi-head self-attention layers.",
    "Knowledge distillation compresses bulky teacher neural architectures into lightweight student models for fast inference.",
    "Quantization downcasts 32-bit floating point weights to 8-bit integers, reducing memory bandwidth on hardware accelerators.",

    # Topic: Cloud Infrastructure & Microservices
    "Kubernetes orchestrates containerized microservices across distributed nodes, ensuring automated scaling and health monitoring.",
    "Docker containers encapsulate operating system dependencies, libraries, and binaries into reproducible deployment images.",
    "Serverless functions execute stateless event-driven compute workloads without requiring continuous server provisioning or management.",
    "API gateways handle client request routing, rate limiting, token authentication, and load balancing across microservices.",
    "Continuous integration and continuous deployment pipelines automate software testing, container building, and staging releases.",
    "Infrastructure as code tools like Terraform manage virtual private clouds, subnets, firewall rules, and compute instances.",
    "Distributed tracing with OpenTelemetry tracks latency, span propagation, and bottleneck anomalies across microservice architectures.",
    "Service meshes manage encrypted mTLS network communications, traffic shifting, and circuit breaker patterns between services.",
    "Kafka message brokers decouple high-throughput streaming events between asynchronous producer and consumer services.",
    "PostgreSQL read replicas scale relational query throughput while Redis in-memory caching reduces repeated database lookups.",

    # Topic: Cybersecurity & Information Security
    "Zero trust network security architecture mandates continuous verification of user identity and endpoint posture before access.",
    "Phishing attacks exploit social engineering techniques to trick employees into divulging sensitive credentials and MFA tokens.",
    "End-to-end encryption secures data in transit and at rest using modern cryptographic algorithms like AES-256 and RSA-4096.",
    "Penetration testing and vulnerability scanning identify misconfigured security groups, SQL injection flaws, and XSS exploits.",
    "Ransomware encrypts critical enterprise storage volumes, demanding cryptocurrency payments for decryption keys.",
    "Web application firewalls inspect HTTP headers and payloads to block distributed denial-of-service and bot traffic.",
    "Security information and event management platforms aggregate audit logs to detect abnormal credential stuffing attempts.",
    "Public key infrastructure relies on trusted certificate authorities to issue, validate, and revoke digital SSL/TLS certificates.",
    "Multi-factor authentication combines biometrics, hardware security keys, and time-based one-time passwords for authentication.",
    "Principle of least privilege ensures IAM service accounts possess only minimal permissions required for their tasks.",

    # Topic: Quantum Computing & Physics
    "Superconducting qubits utilize Josephson junctions operating at millikelvin dilution refrigerator temperatures for quantum operations.",
    "Quantum entanglement establishes non-local correlation between qubits, enabling superdense coding and Bell state teleportation.",
    "Quantum error correction codes protect fragile logical qubits from environmental decoherence and phase-flip gate errors.",
    "Shor's quantum algorithm achieves polynomial time prime factorization, posing risks to traditional public key cryptography.",
    "Grover's search algorithm provides quadratic quantum speedup for searching unsorted databases and NP-hard problem verification.",
    "Quantum circuit transpilation maps abstract quantum gates onto native hardware basis gate sets and coupling connectivity graphs.",
    "Variational quantum eigensolvers approximate molecular ground state energies on noisy intermediate-scale quantum devices.",
    "Trapped-ion quantum computers manipulate ytterbium ions using focused laser beams to execute high-fidelity entangling gates.",
    "Decoherence causes quantum state superpositions to collapse into classical probabilities due to thermal fluctuations.",
    "Quantum key distribution exploits photon polarization states to establish unconditionally secure cryptographic communication links.",

    # Topic: Autonomous Systems & Robotics
    "LiDAR sensors emit pulsed laser beams to construct accurate 3D point cloud maps of surrounding autonomous vehicle environments.",
    "Simultaneous localization and mapping algorithms enable mobile robots to navigate unknown indoor environments without GPS signals.",
    "Kalman filters and particle filters fuse noisy accelerometer, gyroscope, and wheel odometry sensor readings for pose estimation.",
    "Path planning algorithms like A-star and Rapidly-exploring Random Trees compute collision-free trajectories for robotic manipulators.",
    "Inverse kinematics calculates target actuator joint angles required to position an end-effector at specific 3D Cartesian coordinates.",
    "Computer vision algorithms track pedestrian motion trajectories to predict collision risks at urban street intersections.",
    "Robot operating system nodes exchange telemetry messages, sensor scans, and actuator velocity commands over publish-subscribe topics.",
    "Model predictive control computes real-time steering and acceleration inputs while respecting dynamic vehicle stability constraints.",
    "Drone autopilots adjust quadcopter brushless motor RPMs based on PID feedback loops from inertial measurement units.",
    "Tactile robotic gripper arrays measure surface pressure distribution to handle fragile agricultural produce without crushing."
]

CUSTOMER_SUPPORT_DOCUMENTS = [
    # Topic: Billing, Invoicing & Subscriptions
    "I was charged twice on my credit card for the monthly subscription renewal. Please issue an immediate refund.",
    "How can I update my billing payment method and credit card expiration date before the next renewal invoice?",
    "Our accounting department needs official VAT tax invoices for the past three quarterly enterprise billing statements.",
    "My subscription was canceled unexpectedly even though my payment card has sufficient funds and is active.",
    "Can you clarify why there is an unexpected overage fee on our latest cloud usage invoice?",
    "We would like to upgrade from the Starter tier to the Pro enterprise plan with annual billing discount.",
    "Please cancel my automatic renewal and remove my saved payment details from the customer billing portal.",
    "I have not received the prorated refund promised by support after downgrading my team subscription plan.",
    "Where can I find the company tax identification number and formal invoice receipts for business expense filing?",
    "My payment failed with a generic processing error code. Can support retry the transaction manually?",

    # Topic: Account Authentication & Password Recovery
    "I forgot my account password and the password reset email is not arriving in my inbox or spam folder.",
    "I lost access to my authenticator app and cannot enter the two-factor authentication verification code.",
    "Someone attempted unauthorized logins to my profile from an unrecognized IP address. Please lock my account.",
    "How can I change the primary email address linked to my administrator account without losing access?",
    "Single sign-on with Google and Okta is throwing a SAML assertion expired error during user login.",
    "My user account was locked due to too many failed password attempts. Please unlock it immediately.",
    "Can an administrator transfer account ownership and permissions to another team member's email address?",
    "I need backup recovery emergency codes because my mobile phone with the authenticator app was stolen.",
    "The session keeps logging me out every five minutes even when I check the remember this device option.",
    "Verification SMS text message never arrives on my registered international telephone number.",

    # Topic: Shipping, Tracking & Delivery
    "The tracking number indicates package delivered but I have searched my doorstep and found nothing.",
    "My order shipment has been stuck in transit at the regional distribution facility for over one week.",
    "Can I update the delivery shipping address for my package before the courier dispatches the parcel?",
    "The courier attempted delivery without ringing the doorbell and left a missed delivery slip on the gate.",
    "Do you offer expedited next-day express shipping to residential addresses in suburban areas?",
    "The shipping tracking link provided in the confirmation email shows an invalid tracking ID error.",
    "Customs clearance is delaying my international package. Does the customer or seller pay import tariffs?",
    "My package was delivered to the wrong street address according to the GPS delivery confirmation photo.",
    "How long does standard ground shipping take once the warehouse fulfills and ships the item?",
    "The parcel arrived completely damaged and torn open with several ordered items missing from the box.",

    # Topic: Returns, Exchanges & Refunds
    "I would like to return the shoes because the size is too small and exchange them for one size larger.",
    "Please send me a prepaid return shipping label so I can return the defective electronic product for refund.",
    "What is your return policy return window? Is it thirty days from delivery or from the purchase date?",
    "I dropped off the return package at the carrier drop-off location five days ago. When will refund process?",
    "Can I return an item to a physical retail store if I originally bought it online from your website?",
    "The replacement replacement exchange unit sent to me also exhibits the exact same hardware defect.",
    "Can I exchange this gift item for store credit without having the original order purchase receipt?",
    "The refund was processed according to your email, but the funds have not appeared in my bank balance.",
    "Do I need to return the product in its original retail manufacturer packaging and accessory box?",
    "Support promised a warranty replacement unit, but I have not received a tracking number for it.",

    # Topic: Mobile App & Website Technical Issues
    "The mobile app crashes immediately on launch after updating to the latest iOS 18 version.",
    "The shopping cart checkout button is unresponsive and freezes when clicking place order on Safari browser.",
    "Push notifications are not working on Android even though all application notification permissions are enabled.",
    "Page load times are extremely slow and images fail to render with repeated 504 gateway timeout errors.",
    "The search filter dropdown does not filter results properly when searching by product category or price.",
    "Exporting reports to CSV or Excel downloads an empty file with zero bytes on Chrome desktop.",
    "Dark mode theme causes button labels and input text to become invisible white text on white background.",
    "The mobile camera barcode scanner fails to focus or recognize product barcodes in low lighting conditions.",
    "Audio playback stops unexpectedly whenever the screen locks or when switching to another background app.",
    "WebSocket disconnected error keeps popping up in the bottom corner of the web dashboard interface."
]

HEALTH_SCIENCE_DOCUMENTS = [
    # Topic: Cardiovascular Health & Cardiology
    "Regular aerobic exercise strengthens myocardial contractility and helps lower resting systolic blood pressure.",
    "Atherosclerosis develops when low-density lipoprotein cholesterol plaques accumulate within coronary artery walls.",
    "Electrocardiograms detect cardiac arrhythmias such as atrial fibrillation and ventricular tachycardia.",
    "Hypertension is a major preventable risk factor for congestive heart failure and ischemic stroke.",
    "Coronary angioplasty and stent placement restore vital oxygenated blood flow to ischemic myocardium.",
    "Echocardiography evaluates cardiac ejection fraction, ventricular wall motion, and heart valve functionality.",
    "Beta-blockers and ACE inhibitors reduce cardiac workload and manage chronic cardiovascular diseases.",
    "Cardiovascular endurance training stimulates collateral angiogenesis and improves systemic vascular compliance.",
    "High plasma troponin biomarker levels provide critical diagnostic evidence of acute myocardial infarction.",
    "Dietary reduction of saturated fats and sodium promotes arterial elasticity and cardiac longevity.",

    # Topic: Immunology & Vaccine Technology
    "Messenger RNA vaccines instruct host ribosomes to translate viral spike glycoproteins to stimulate antibodies.",
    "Adaptive immunity relies on memory B lymphocytes to produce neutralizing antibodies against pathogen antigens.",
    "Cytotoxic T cells identify and destroy pathogen-infected cells expressing foreign peptide fragments on MHC-I.",
    "Adjuvants in vaccine formulations boost antigen presentation and enhance innate immune receptor activation.",
    "Autoimmune disorders occur when the immune system mistakenly targets self-antigens and healthy tissue.",
    "Herd immunity thresholds prevent widespread infectious disease outbreaks within vulnerable populations.",
    "Monoclonal antibody therapies neutralize specific inflammatory cytokines in severe immune dysregulation.",
    "Mast cells degranulate and release histamine during acute allergic hypersensitivity reactions.",
    "Phagocytes including macrophages and neutrophils engulf and digest invading bacterial pathogens via phagocytosis.",
    "Immunological memory enables rapid and potent antibody titer surges upon secondary antigen exposure.",

    # Topic: Neuroscience & Brain Health
    "Synaptic plasticity through long-term potentiation forms the neurobiological foundation of learning and memory.",
    "Neurotransmitters such as dopamine and serotonin modulate mood, reward processing, and cognitive motivation.",
    "The blood-brain barrier shields sensitive neuronal networks from circulating toxins, pathogens, and cytokines.",
    "Neurodegenerative diseases like Alzheimer's involve abnormal amyloid-beta plaques and hyperphosphorylated tau tangles.",
    "Functional magnetic resonance imaging measures hemodynamic blood oxygen level changes during cognitive tasks.",
    "Slow-wave deep sleep facilitates glymphatic clearance of metabolic waste products from brain tissue.",
    "Myelin sheaths formed by oligodendrocytes insulate axonal fibers to accelerate action potential conduction velocities.",
    "Neurogenesis in the adult hippocampal subgranular zone supports continuous spatial learning and emotional regulation.",
    "Chronic psychological stress elevates cortisol levels, causing dendritic atrophy in prefrontal cortical circuits.",
    "Circadian rhythms regulated by the suprachiasmatic nucleus synchronize melatonin release and sleep-wake cycles.",

    # Topic: Nutrition, Metabolic Health & Gut Microbiome
    "The human gut microbiome digests complex prebiotic dietary fibers to synthesize beneficial short-chain fatty acids.",
    "Insulin resistance in skeletal muscle and adipose tissue underlies the pathogenesis of type 2 diabetes mellitus.",
    "A diverse intestinal bacterial flora supports mucosal barrier integrity and modulates systemic inflammatory tone.",
    "Intermittent fasting activates cellular autophagy and enhances mitochondrial biogenesis in metabolic tissues.",
    "Probiotic fermented foods containing Lactobacillus strains promote gastrointestinal digestion and immune homeostasis.",
    "Excess consumption of ultra-processed carbohydrates and high-fructose corn syrup promotes non-alcoholic fatty liver disease.",
    "Omega-3 polyunsaturated fatty acids like EPA and DHA exert potent anti-inflammatory effects on cell membranes.",
    "Metabolic syndrome is clinically characterized by visceral obesity, hyperglycemia, dyslipidemia, and elevated blood pressure.",
    "The gut-brain axis mediates bi-directional neural communication between intestinal microbiota and the central nervous system.",
    "Micronutrients including vitamin D, magnesium, and zinc act as vital cofactors for hundreds of enzymatic metabolic reactions.",

    # Topic: Climate Science & Renewable Energy
    "Anthropogenic greenhouse gas emissions trap infrared radiation in the atmosphere, accelerating global climate change.",
    "Photovoltaic solar panels convert incident solar irradiance directly into electricity via the semiconductor photoelectric effect.",
    "Offshore wind turbines harness high-velocity ocean wind currents to drive electromagnetic generators for grid power.",
    "Ocean acidification occurs as seawater absorbs excess atmospheric carbon dioxide, forming carbonic acid that lowers pH.",
    "Lithium-ion and solid-state batteries provide grid-scale energy storage to buffer intermittent renewable solar and wind power.",
    "Retreating polar ice sheets and mountain glaciers decrease Earth's albedo, exacerbating planetary thermal absorption.",
    "Green hydrogen produced through water electrolysis using renewable electricity serves as zero-carbon industrial fuel.",
    "Carbon capture and geological sequestration technologies isolate point-source industrial carbon dioxide emissions.",
    "Extreme weather events including prolonged droughts, megafires, and Category 5 hurricanes are increasing in frequency.",
    "Transitioning from fossil fuels to decentralized smart power grids reduces atmospheric particulate pollution and sulfur dioxide."
]


SAMPLE_DATASETS = {
    "AI & Technology": {
        "description": "50 technical documents across Deep Learning, Cloud Microservices, Cybersecurity, Quantum Computing, and Autonomous Robotics.",
        "documents": TECH_AI_DOCUMENTS,
        "suggested_k": 5
    },
    "Customer Support Tickets": {
        "description": "50 user support tickets covering Billing/Invoicing, Account Auth/SSO, Shipping/Tracking, Returns/Refunds, and Mobile App Bugs.",
        "documents": CUSTOMER_SUPPORT_DOCUMENTS,
        "suggested_k": 5
    },
    "Health & Environmental Science": {
        "description": "50 scientific documents covering Cardiology, Immunology/Vaccines, Neuroscience, Gut Microbiome/Nutrition, and Renewable Energy/Climate.",
        "documents": HEALTH_SCIENCE_DOCUMENTS,
        "suggested_k": 5
    }
}


def get_sample_dataframe(dataset_name: str) -> pd.DataFrame:
    """Return sample dataset as a pandas DataFrame."""
    if dataset_name not in SAMPLE_DATASETS:
        dataset_name = "AI & Technology"
    data = SAMPLE_DATASETS[dataset_name]
    df = pd.DataFrame({
        "doc_id": range(1, len(data["documents"]) + 1),
        "text": data["documents"]
    })
    return df
