FROM pytorch/pytorch
COPY jointclassifier /app/jointclassifier
COPY requirements.txt /app
COPY ml_server.py /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5001
ENTRYPOINT ["python"]
CMD ["ml_server.py"]