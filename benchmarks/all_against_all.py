from deepface import DeepFace

INPUT_PATHS = "../utils/file_list_short_short.txt"

THRESHOLDS = [0.2, 0.3]

METRICS = ["cosine", "euclidean", "euclidean_l2"]

model = DeepFace.build_model('VGG-Face')

def verify(distance_score, threshold):
    return distance_score <= threshold

with open(INPUT_PATHS) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

genuine_acceptances = dict()
genuine_rejections = dict()
false_acceptances = dict()
false_rejections = dict()

genuine_attempts = dict()
impostor_attempts = dict()

def perform_all_against_all():

    for metric in METRICS:

        genuine_acceptances[metric] = dict()
        genuine_rejections[metric] = dict()
        false_acceptances[metric] = dict()
        false_rejections[metric] = dict()

        genuine_attempts[metric] = dict()
        impostor_attempts[metric] = dict()

        for threshold in THRESHOLDS:
            threshold_str = str(threshold)

            genuine_acceptances[metric][threshold_str] = 0
            genuine_rejections[metric][threshold_str] = 0
            false_acceptances[metric][threshold_str] = 0
            false_rejections[metric][threshold_str] = 0

            genuine_attempts[metric][threshold_str] = 0
            impostor_attempts[metric][threshold_str] = 0

            for first_identity_index in range(0, len(lines)):
                first_identity_name = lines[first_identity_index].split('/')[3]

                for second_identity_index in range(0, len(lines)):

                    if (second_identity_index == first_identity_index): continue

                    second_identity_name = lines[second_identity_index].split('/')[3]

                    if (first_identity_name == second_identity_name):
                        genuine_attempts[metric][threshold_str] += 1
                    else:
                        impostor_attempts[metric][threshold_str] += 1

                    result = DeepFace.verify(
                        img1_path = lines[first_identity_index],
                        img2_path = lines[second_identity_index],
                        model = model
                    )

                    verified = verify(result['distance'], threshold)

                    genuine_acceptances[metric][threshold_str] += int(verified and (first_identity_name == second_identity_name))

                    genuine_rejections[metric][threshold_str] += int(not verified and not (first_identity_name == second_identity_name))

                    false_acceptances[metric][threshold_str] += int(verified and not (first_identity_name == second_identity_name))

                    false_rejections[metric][threshold_str] += int(not verified and (first_identity_name == second_identity_name))

                    print("matching faces: ", first_identity_index, ", ", first_identity_name, " VS. ", second_identity_index, ", ", second_identity_name)
                    print("DeepFace says: ", result['verified'])
                    print("Should actually be: ", result['verified'] and (first_identity_name == second_identity_name))
                    print("Distance: ", result['distance'], ", threshold: ", threshold)
                    print()

def print_recognition_metrics():
    for metric in METRICS:

        for threshold in THRESHOLDS:

            threshold_str = str(threshold)

            # print("GA[", threshold_str, "]: ", genuine_acceptances[metric][threshold_str], genuine_attempts[metric][threshold_str])
            # print("GR[", threshold_str, "]: ", genuine_rejections[metric][threshold_str], genuine_attempts[metric][threshold_str])
            # print("FA[", threshold_str, "]: ", false_acceptances[metric][threshold_str], impostor_attempts[metric][threshold_str])
            # print("FR[", threshold_str, "]: ", false_rejections[metric][threshold_str], impostor_attempts[metric][threshold_str])

            print("-----------")

            print(
                "Genuine attemps[", metric, "][", threshold_str, "]: ",
                genuine_attempts[metric][threshold_str]
            )
            print(
                "Impostor attemps[", metric, "][", threshold_str, "]: ",
                impostor_attempts[metric][threshold_str]
            )
            print()
            print(
                "GAR[", metric, "][", threshold_str, "]: ",
                genuine_acceptances[metric][threshold_str] / genuine_attempts[metric][threshold_str]
            )
            print(
                "GRR[", metric, "][", threshold_str, "]: ",
                genuine_rejections[metric][threshold_str] / impostor_attempts[metric][threshold_str]
            )
            print(
                "FAR[", metric, "][", threshold_str, "]: ",
                false_acceptances[metric][threshold_str] / impostor_attempts[metric][threshold_str]
            )
            print(
                "FRR[", metric, "][", threshold_str, "]: ",
                false_rejections[metric][threshold_str] / genuine_attempts[metric][threshold_str]
            )
            print(
                "Error rate[", metric, "][", threshold_str, "]: ",
                (
                    false_acceptances[metric][threshold_str] +
                    false_rejections[metric][threshold_str]
                ) / (
                    genuine_attempts[metric][threshold_str] +
                    impostor_attempts[metric][threshold_str]
                )
            )
            print()


perform_all_against_all()

print_recognition_metrics()

# TODO: dump dicts to json file
