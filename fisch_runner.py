import fisch

#Edit autohotkey script if you want to change click position.
#Edit fisch script if you want to change q-learning or control bar threshold variables

#click delay of throw in seconds, and optional delay
param_throw = [0.45, 2]
#Image of the "shake" circle, delay after a click, threshold for the image, optional initial delay
param_shake = ['train data/shake_circle.png', 1.5, 0.3]
#fish image, threshold, pixels added as padding for the fish image, amount of allowed non found fishes
param_image = ['train data/fish.png', 0.855, 100, 5]
#number of minimum allowed pixel gaps in a control bar sequence and control bar error gaps
param_sequence = [20, 7]
#IMPORTANT!!! (Based on your screen x and y coordinates)
#first and last x coordinate of reel bar, y coordinate of reel bar and added height for the fish image detection
param_coords = [572, 1347, 853, 153]

#Amount of games that are going to play
number_of_games = 9999

#Usage
for i in range(number_of_games):
    fisch.throw(param_throw[0])
    fisch.shake(param_shake[0], param_shake[1], param_shake[2])
    fisch.reel(param_coords, param_image, param_sequence)

