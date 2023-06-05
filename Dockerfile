# syntax=docker/dockerfile:1

FROM python:3.11
WORKDIR /project
COPY . .
RUN sh ./scripts/install_requirements.sh  
EXPOSE 8001 8002
RUN 
CMD ["sh", "scripts/run_services.sh"]