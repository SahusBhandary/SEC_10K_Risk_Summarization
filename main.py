from datetime import date

def main():

    # Validate Args
    while True:
        # Input ticker and year range
        ticker = input("Please enter a ticker (ex: AAPL): ")
        year_range = input("Please enter a year range (ex: 2020-2024) [The earliest year is 2001 and the latest year is 2024]: ")

        # Check if the ticker contains numbers and if it has a length of 4
        if not (1 <= len(ticker) <= 5 and ticker.isalpha()):
            print("Invalid Ticker")
            continue
        
        start_year, end_year = "", ""
        # Split year range
        try:
            tokens = year_range.split("-")
            start_year, end_year = tokens[0], tokens[1]
        except Exception as e:
            print("Invalid Year Range Format (ex: 2020-2024)")
            continue
        
        # Check if the years entered are valid
        if not (start_year and end_year and int(start_year) >= 2001 and int(end_year) <= 2024):
            print("Invalid year range, please enter a year from 2001 to 2024!")
            continue

        break


        


        

if __name__ == "__main__":
    main()
        

