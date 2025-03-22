# Proyecto YOLO con Docker

Este proyecto permite ejecutar un modelo YOLO (You Only Look Once) para detección de objetos en tiempo real utilizando Docker. El sistema se levanta con un solo comando y aprovecha la GPU si está disponible para acelerar la inferencia.

## Requisitos

- **Docker** y **Docker Compose** instalados.


## Instalación

1. Clonar el repositorio:
```sh
git clone https://github.com/anitachan/yolo
cd proyecto-yolo
```

2. Ejecutar el proyecto:
```sh
docker compose up
```


Tras ejecutar docker compose up, accede a la interfaz web en:
```
http://localhost/
```

Conceder permisos de cámara

