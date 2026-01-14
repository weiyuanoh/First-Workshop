import pandas as pd 
from ib_insync import IB 


ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)
accounts = ib.managedAccounts()  # Typically returns a list of account IDs
account_id = accounts[0]         # If you only have one account

print(accounts)