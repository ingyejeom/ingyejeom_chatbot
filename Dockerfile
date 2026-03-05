# 1. 베이스 이미지: 파이썬 버전
FROM python:3.11-slim

# 2. 작업 디렉토리 설정 (이 안에서 모든 작업이 이루어짐)
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (ChromaDB 등이 내부적으로 C++ 컴파일러를 요구할 수 있음)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 패키지 설치 (코드보다 먼저 복사해서 캐시 효율을 높임)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 내 프로젝트 소스코드 전체 복사 (.dockerignore에 적힌 건 빼고 복사됨)
COPY . .

# 6. FastAPI가 사용할 포트 번호 명시
EXPOSE 8000

# 7. 컨테이너가 켜질 때 실행할 명령어 (main.py의 app 객체를 uvicorn으로 실행)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]