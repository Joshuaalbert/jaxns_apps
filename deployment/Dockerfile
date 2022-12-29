FROM python:3.10.8


WORKDIR /code

# requirements rarely change
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./global_opt/requirements.txt /code/global_opt/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/global_opt/requirements.txt
COPY ./global_opt /code/global_opt
RUN pip install /code/global_opt

COPY ./bayes_opt/requirements.txt /code/bayes_opt/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/bayes_opt/requirements.txt
COPY ./bayes_opt /code/bayes_opt
RUN pip install /code/bayes_opt

COPY ./fourier/requirements.txt /code/fourier/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/fourier/requirements.txt
COPY ./fourier /code/fourier
RUN pip install /code/fourier

COPY ./gaussian_processes/requirements.txt /code/gaussian_processes/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/gaussian_processes/requirements.txt
COPY ./gaussian_processes /code/gaussian_processes
RUN pip install /code/gaussian_processes


ENTRYPOINT ["pytest", "-s", "/code"]