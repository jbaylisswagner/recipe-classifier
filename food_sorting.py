"""
sorting foods and outputting them to a json file
"""

import json

def main():

  #make a dictionary
  food_dict = {}
  with open("food_groups.json", 'r') as food_file:
      food_dict = json.load(food_file)
  print("prevous food dict: ", food_dict)

  with open("train.json", "r") as read_file:
      data = json.load(read_file)
  
  print("\n\n--------Sort ingredients into categories----------")
  print("Categories: spices, proteins, veggies, fruits, dairy products, grains, oils")
  print("We will abbreviate these as S, P, V, F, D, G, and O.")
  print("Enter the number that corresponds to the food's category.\n\n")
  flag = False
  for recipe in data:
      if flag == False: 
        ingredients_lst = recipe['ingredients']
        for ingredient in ingredients_lst:
          
          if flag == False and ingredient not in food_dict:
            print("\n\n\nFood: ", ingredient, "\n")
            print("(1) Spices,  (2) Proteins, (3) Veggies, (4) Fruits, (5) Dairy, (6) Grains, (7) Oils, (8) Other, OR (q) to quit")
            category = input("\nEnter a number: ")
            if category == 'q':
              flag = True

            while category not in ['1', '2', '3', '4', '5','6', '7', '8', 'q']:
              print("Invalid input! Try again.")
              category = input("Enter a number: ")
            
            if category == '1':
              food_dict[ingredient] = 'spice'

            elif category == '2':
              food_dict[ingredient] = 'protein'

            elif category == '3':
              food_dict[ingredient] = 'vegetable'

            elif category == '4':
              food_dict[ingredient] = 'fruit'

            elif category == '5':
              food_dict[ingredient] = 'dairy'

            elif category == '6':
              food_dict[ingredient] = 'grain'

            elif category == '7':
              food_dict[ingredient] = 'oil'

  print(food_dict)
  

  #dump into json file 
  with open('food_groups.json', 'w') as fp:
      json.dump(food_dict, fp)



  
main()
