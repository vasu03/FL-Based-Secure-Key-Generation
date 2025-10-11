# Federated Learning Based Secure Key Generation

## Overview
- Federated Learning based decentralized secure key generation for IoT networks. 
- The project aims to develop a privacy-preserving and decentralized key generation mechanism for IoT (Internet of Things) networks using Federated Learning (FL). 
- Traditional key generation schemes rely on centralized servers, which pose security and privacy risks.
- This project explores how FL can enable multiple IoT devices to collaboratively train a machine learning model for key generation without sharing raw data, thereby ensuring data confidentiality, resilience against attacks, and scalability.
- Currently the project is being developed under a simulated IoT network environment, and no physical IoT devices are being used.

## Setup & Installation

1. Clone the official repository:
    ```
    git clone https://github.com/vasu03/FL-Based-Secure-Key-Generation.git
    cd FL-Based-Secure-Key-Generation/
    ```

2. Install the required dependencies as per your operating system:
    - `Windows OS`
        ```
        pip install -r requirements.txt
        ```

    - `Linux or MacOS`
        ```
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        deactivate
        ```
> [NOTE]: You must have latest version of `pip` installed on your system

## Testing & Execution

- `Windows OS`
    ```
    python3 main.py
    ```

- `Linux or MacOS`
    ```
    source venv/bin/activate
    python3 main.py
    deactivate
    ```

---

## License & Disclaimer

> **Notice:**  
> This project is developed **solely for educational and academic research purposes**.  
> It is **not intended or recommended for real-world, commercial, or production use**.  
> The author(s) do **not endorse or take responsibility** for any misuse or unethical application of this code.  
> Use responsibly and in compliance with all applicable laws.

This project is licensed under the **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**.  
You are free to use, modify, and distribute this software under the terms of that license.
