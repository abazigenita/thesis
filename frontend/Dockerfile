ARG BUILDPLATFORM=linux/amd64

FROM --platform=${BUILDPLATFORM} node:latest as build

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm install

COPY . ./

RUN npm run build

FROM --platform=${BUILDPLATFORM} nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]