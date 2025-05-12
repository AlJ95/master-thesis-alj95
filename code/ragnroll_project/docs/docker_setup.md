# Docker Setup for RAGnRoll Project

This document describes how to set up and run the RAGnRoll project services using Docker Compose.

## Prerequisites

*   Docker: [Install Docker](https://docs.docker.com/get-docker/)
*   Docker Compose: Usually included with Docker Desktop. Verify with `docker compose version`.

## Configuration

The project uses Docker Compose files (`docker-compose.yml`, `docker-compose-vdbs.yml`) and an environment file (`.env.local`) for configuration, especially for secrets.

1.  **Copy `.env.local` file to `.env`:**
    In the root directory of the `ragnroll_project` (`code/ragnroll_project/`), copy the `.env.local` file to `.env`.

2.  **Change `.env`:**
    Copy the following content into `.env`. **Crucially, replace all placeholder values marked with `# CHANGEME` with your actual secrets.**


## Running the Services

You can run the core services and the vector database services separately or together.

1.  **Navigate to the project directory:**
    Open your terminal and change the directory to `code/ragnroll_project`.

    ```bash
    cd path/to/code/ragnroll_project
    ```

2.  **Start Core Services:**
    This starts Langfuse, MLflow, Postgres, ClickHouse, Minio, and Redis.

    ```bash
    docker compose up -d
    ```
    *   The `-d` flag runs the containers in detached mode (in the background).

3.  **Start Vector Databases (Optional):**
    If you need ChromaDB or Qdrant, run:

    ```bash
    docker compose -f docker-compose-vdbs.yml up -d
    ```

## Accessing Services

Once the containers are running, you can access the services via your web browser:

*   **Langfuse UI:** [http://localhost:3000](http://localhost:3000)
*   **MLflow UI:** [http://localhost:8080](http://localhost:8080)

*Note: Other services like Postgres, Redis, and ClickHouse are primarily accessed internally by other services or via specific database clients connected to their respective localhost ports (e.g., 5432 for Postgres, 6379 for Redis, 8123/9000 for ClickHouse).*

## Stopping Services

To stop the running containers:

1.  **Stop Core Services:**
    Make sure you are in the `code/ragnroll_project` directory.

    ```bash
    docker compose down
    ```

2.  **Stop Vector Databases:**

    ```bash
    docker compose -f docker-compose-vdbs.yml down
    ```

    The `down` command stops and removes the containers, networks, and potentially volumes (unless specified otherwise).
