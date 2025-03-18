FROM node:18

LABEL maintainer="frontenderp"
WORKDIR /frontend

COPY package*.json ./
RUN npm install

COPY ./app ./

EXPOSE 5000


CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5000"]
