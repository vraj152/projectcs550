import content_based_recsys
user_Id = input("Enter User Id:")
weight_dict, cosine_distances, weight_list = content_based_recsys.build_user_profile(user_Id)
count = 1
print("Recommendations to user: ",user_Id)

for key, value in cosine_distances.items():
    if(count<20):
        print("Movie with ID: %d with similarity %f" % (key, value))
        count = count + 1
    else:
        break