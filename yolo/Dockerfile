# Stage 1: Build Angular App
FROM node:16 as builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build --prod

# Stage 2: Serve the Angular Build with NGINX
FROM nginx:alpine
# Adjust "yolo" below if your Angular build output folder differs (e.g., dist/your-angular-app)
COPY --from=builder /app/dist/yolo /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
