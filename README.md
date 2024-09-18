# NL2SQL-Agent

![NL2SQL-Agent](https://github.com/mohammed97ashraf/NL2SQL-Agent/blob/main/Untitled%20design.jpg?raw=true)

## Setting Up the Database

To follow along with this project, we'll be using the Pagila sample database, which is a PostgreSQL-based database designed for learning and testing SQL queries. To get started, follow these steps to set up the database environment:

1. **Clone the Repository**: First, clone the Pagila repository from GitHub. You can do this by running the following command:

    ```bash
    git clone https://github.com/devrimgunduz/pagila
    ```

2. **Set Up the Database Using Docker**: Once you've cloned the repository, navigate into the directory and use Docker Compose to quickly set up the PostgreSQL database:

    ```bash
    cd pagila
    docker-compose up -d
    ```

3. **Verify the Database**: After the container is up and running, you can verify that the database is working correctly by connecting to it using any PostgreSQL client, such as `psql` or a graphical tool like pgAdmin:

    ```bash
    docker exec -it pagila_postgres_1 psql -U postgres -d pagila
    ```

## To Run this

1. **Setup your API keys in `.env` file**:
   ```bash
   LANGCHAIN_API_KEY="lsv2---------"
   OPENAI_API_KEY="sk------"
   LANGCHAIN_TRACING_V2="true"
   LANGCHAIN_PROJECT="---"
   GROQ_API_KEY="gsk-----"
   ```
2. **Install the dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```
   python app.py
   ```
