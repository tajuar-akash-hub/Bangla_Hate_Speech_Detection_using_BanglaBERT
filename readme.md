```markdown
# Bangla Hate Speech Detection using BanglaBERT

This project provides a machine learning solution for detecting hate speech in Bangla text using the **BanglaBERT** model.  
It includes APIs built with **FastAPI**, and a simple **Streamlit** interface for testing.  
The application is containerized with **Docker** for easy deployment.

---

## Features
- Hate speech detection for Bangla text
- REST API built with FastAPI
- Interactive UI using Streamlit
- Dockerized for easy deployment
- Pretrained BanglaBERT model integration

---

## Project Structure
```

.
├── app/                  # FastAPI app code
├── streamlit\_app/        # Streamlit interface
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build instructions
└── README.md             # Documentation

````

---

## Installation

### Prerequisites
- Python 3.9+  
- Docker (if running via container)  
- Git  

### Clone Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
````

### Local Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

### Option 1: Run Locally

**FastAPI Server**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:

```
http://127.0.0.1:8000/docs
```

**Streamlit App**

```bash
streamlit run streamlit_app/main.py
```

---

### Option 2: Run with Docker

**Build the Image**

```bash
docker build -t blacksllm/hate:latest .
```

**Run the Container**

```bash
docker run -d -p 8000:8000 blacksllm/hate:latest
```

Check the API at:

```
http://localhost:8000/docs
```

---

## Deploy on AWS EC2

1. Launch an Ubuntu EC2 instance
2. Install Docker:

   ```bash
   sudo apt update && sudo apt install docker.io -y
   ```
3. Login and pull image:

   ```bash
   sudo docker login
   sudo docker pull blacksllm/hate:latest
   ```
4. Run:

   ```bash
   sudo docker run -d -p 8000:8000 blacksllm/hate:latest
   ```

Access API via:

```
http://<ec2-public-ip>:8000/docs
```

---

## API Endpoints

* `POST /predict` → Predict hate speech in Bangla text

Example:

```json
{
  "text": "তুমি খারাপ"
}
```

Response:

```json
{
  "prediction": "Hate Speech"
}
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.

---

## License

[MIT](LICENSE)

```

---

```

