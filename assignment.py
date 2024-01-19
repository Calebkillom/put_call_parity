#!/usr/bin/python3
""" Stochastic Calculus assignment """
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
import statsmodels.api as sm
import sys

url = (
    "https://www.marketwatch.com/investing/stock/"
    "xom/options?mod=mw_quote_tab"
)

page = requests.get(url)

soup = BeautifulSoup(page.content, "html.parser")

""" Find all table bodies with class "table__body" """
table_bodies = soup.find_all("tbody", class_="table__body")

""" Initialize a list to store all option data in one list """
options_data = []

""" Iterate over each table body """
for table_body in table_bodies:
    """ Find all table rows within the current table body """
    rows = table_body.find_all("tr")

    """ Initialize variables to track the type of row """
    current_price_row = False
    options_header_row = False

    """ Iterate over each table row """
    for row in rows:
        """ Check if the row is a current price row """
        if "current-price" in row.get("class", []):
            current_price_row = True
            options_header_row = False
        elif "table__row" in row.get(
                "class", []) and "current-price" not in row.get("class", []):
            options_header_row = True
            current_price_row = False
        else:
            options_header_row = False
            current_price_row = False

        """
        Initialize dictionaries for call,
        strike, and put options for each row
        """
        call_option_data = {}
        strike_data = {}
        put_option_data = {}

        """
        Extract data from specific columns
        within each row for call options
        """
        call_data_elems = row.find_all(
                "div", class_="option__cell in-money")

        """
        Check if elements are found before accessing
        their text attribute for call options
        """
        if len(call_data_elems) >= 3:
            call_bid_price = call_data_elems[1].text.strip()
            call_ask_price = call_data_elems[2].text.strip()

            call_option_data = {
                "Type": "Call",
                "Bid Price": call_bid_price,
                "Ask Price": call_ask_price
            }

        """
        Extract data from specific columns
        within each row for put options
        """
        put_data_elems = row.find_all("div", class_="option__cell")

        if len(put_data_elems) >= 4:
            put_bid_price = put_data_elems[-4].text.strip()
            put_ask_price = put_data_elems[-3].text.strip()

            put_option_data = {
                "Type": "Put",
                "Bid Price": put_bid_price,
                "Ask Price": put_ask_price
            }

        """
        Check if elements are found before
        accessing their text attribute for strike
        """
        strike_price_elem = row.find("div", class_="option__cell strike")

        if strike_price_elem:
            strike_price = strike_price_elem.text.strip()
            strike_data = {
                "Type": "Strike",
                "Strike Price": strike_price
            }

        """
        Append dictionaries to the list
        for each row, organized by row type
        """
        if current_price_row:
            options_data.append({"Row Type": "Current Price", "Data":
                                 {"Strike Data": strike_data}})
        elif options_header_row:
            options_data.append({"Row Type": "Options Header", "Data":
                                 {"Call Options": call_option_data,
                                  "Strike Data": strike_data,
                                  "Put Options": put_option_data}})
        else:
            options_data.append({"Row Type": "Unknown", "Data": {}})


def calculate_midpoint(bid_price, ask_price):
    """
    Calculate the midpoint between bid and ask prices.
    """
    bid = float(bid_price)
    ask = float(ask_price)
    midpoint = (bid + ask) / 2
    return round(midpoint, 3)


""" Initialize a list to store new data """
midpoint_data = []

""" Iterate through options_data """
for option_data in options_data:
    if option_data["Row Type"] == "Options Header":
        """ Extract relevant data """
        call_bid_price = option_data["Data"]["Call Options"]["Bid Price"]
        call_ask_price = option_data["Data"]["Call Options"]["Ask Price"]
        put_bid_price = option_data["Data"]["Put Options"]["Bid Price"]
        put_ask_price = option_data["Data"]["Put Options"]["Ask Price"]
        strike_price = option_data["Data"]["Strike Data"]["Strike Price"]

        """ Calculate midpoints """
        call_midpoint = calculate_midpoint(call_bid_price, call_ask_price)
        put_midpoint = calculate_midpoint(put_bid_price, put_ask_price)

        """ Create a new dictionary with the required information """
        new_data = {
            "Strike Price": strike_price,
            "Call Midpoint": call_midpoint,
            "Put Midpoint": put_midpoint
        }

        """ Append to the list """
        midpoint_data.append(new_data)

""" Print the new data """
for data in midpoint_data:
    print(data)


""" plotting the data """
"""Extracting data for plotting """
strike_prices_call = np.array(
        [float(data["Strike Price"]) for data in midpoint_data
         if data["Call Midpoint"] != 0])

strike_prices_put = np.array(
        [float(data["Strike Price"]) for data in midpoint_data
         if data["Put Midpoint"] != 0])

call_midpoints = np.array(
        [float(data["Call Midpoint"]) for data in midpoint_data
         if data["Call Midpoint"] != 0])

put_midpoints = np.array(
    [float(data["Put Midpoint"]) for data in midpoint_data
     if data["Put Midpoint"] != 0])

""" Sort the data by strike price """
sort_order_call = np.argsort(strike_prices_call)
sort_order_put = np.argsort(strike_prices_put)

strike_prices_call = strike_prices_call[sort_order_call]
strike_prices_put = strike_prices_put[sort_order_put]
call_midpoints = call_midpoints[sort_order_call]
put_midpoints = put_midpoints[sort_order_put]

""" Interpolate for smooth curves """
smooth_strike_call = np.linspace(min(strike_prices_call),
                                 max(strike_prices_call), 100)
smooth_call = np.interp(smooth_strike_call, strike_prices_call, call_midpoints)

smooth_strike_put = np.linspace(
            min(strike_prices_put), max(
                strike_prices_put), 100)
smooth_put = np.interp(smooth_strike_put, strike_prices_put, put_midpoints)

""" Plotting using matplotlib and numpy """
plt.figure(figsize=(10, 6))
plt.plot(smooth_strike_call, smooth_call,
         label='Call Midpoint', linestyle='-', color='blue')

plt.plot(smooth_strike_put, smooth_put,
         label='Put Midpoint', linestyle='-', color='red')

plt.scatter(strike_prices_call, call_midpoints, color='blue')
plt.scatter(strike_prices_put, put_midpoints, color='red')


""" Plotting the line for the strike price """
"""
Check if elements are found before accessing
their text attribute for current strike price
"""
current_strike_price_elem = soup.find(
    "div", class_="option__cell strike current l-hide")

"""
Check if elements are found before accessing
their text attribute for current strike price
"""
if current_strike_price_elem:
    current_strike_price = float(current_strike_price_elem.text.strip())
else:
    current_strike_price = 0.0

""" Plotting the line for the strike price """
plt.axvline(x=current_strike_price, color='green',
            linestyle='--', label='Current Strike Price')

plt.title('Call and Put Midpoints vs. Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Payoff')
plt.legend()
plt.grid(True)
plt.savefig('midpoint7_plot.png')

"""Linear Regression"""

""" Convert midpoint_data to a DataFrame """
df = pd.DataFrame(midpoint_data)

""" Extract relevant columns """
df = df[df["Call Midpoint"] != 0]
df["Strike Price"] = pd.to_numeric(df["Strike Price"])
df["Call Midpoint"] = pd.to_numeric(df["Call Midpoint"])
df["Put Midpoint"] = pd.to_numeric(df["Put Midpoint"])

""" Add a constant term to the independent variables """
df = sm.add_constant(df)

""" Define independent and dependent variables """
X = df[["const", "Put Midpoint", "Strike Price"]]
y = df["Call Midpoint"]

""" Fit the linear regression model """
model = sm.OLS(y, X).fit()

""" Display summary statistics """
print(model.summary())

""" Save the current standard output for later restoration """
original_stdout = sys.stdout

""" Redirect the standard output to a file """
with open('ols_summary.txt', 'w') as file:
    sys.stdout = file
    print(model.summary())

""" Restore the original standard output """
sys.stdout = original_stdout
