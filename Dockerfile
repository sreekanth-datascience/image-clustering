FROM tensorflow/tensorflow
COPY . /usr/app/
EXPOSE 5000
workdir /usr/app/
RUN pip install -r requirements.txt
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD python uploadfiles.py
