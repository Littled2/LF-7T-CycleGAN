<?php

    require_once __DIR__ . "/../../database/demand-db.php";
    require_once __DIR__ . "/../../database/demand-files.php";
    require_once __DIR__ . "/../../helpers/helpers.php";

    must_be_authenticated();

    safely_start_session();

    http_method_must_be("GET");

    header("Content-Type: application/json");

    $database = new DemandDB();

    $responses = $database->get_documents("mri-survey-responses")->documents;

    $categories = array(
        "artefacts",
        "structure",
        "snr",
        "overall"
    );

    $overall = array();

    // Calculate the votes per category, per model
    for ($i=0; $i < count($responses); $i++) { 
        $response = $responses[$i];

        for ($j=0; $j < count($categories); $j++) { 
        
            $category = $categories[$j];

            if(!isset($response[$category])) {
                continue;
            }

            $winningModel = $response[$category];

            if(!isset($overall[$winningModel])) {
                $overall[$winningModel] = array();
            }

            if(!isset($overall[$winningModel][$category])) {
                $overall[$winningModel][$category] = 0;
            }

            $overall[$winningModel][$category] += 1;
        }
    }

    // Calculate the total number of votes submitted per category
    $category_totals = array();
    foreach ($categories as $category) {
        $total = 0;
        foreach ($overall as $model_categories) {
            if (isset($model_categories[$category])) {
                $total += $model_categories[$category];
            }
        }
        $category_totals[$category] = $total;
    }

    // Combine analytics for table
    $analytics = array();
    foreach ($overall as $model => $model_categories) {
        $analytics[$model] = array();
        foreach ($categories as $category) {
            $votes = isset($model_categories[$category]) ? $model_categories[$category] : 0;
            $total = $category_totals[$category];
            $analytics[$model][$category] = array(
                "votes" => $votes,
                "total"=> $total,
                "percentage" => $total > 0 ? round(($votes / $total) * 100, 1) : 0.0 // calculate percentage
            );
        }
    }

    // Build a map of survey participants
    $participants_docs = $database->get_documents("mri-surveys")->documents;
    $participants = array();

    foreach ($participants_docs as $participant) {

        $participants[$participant["id"]] = array(
            "surveyId" => $participant["id"],
            "name" => $participant["name"],
            "email" => $participant["email"],
            "jobTitle" => $participant["jobTitle"]
        );
    }

    // Read feedback from database
    $feedback_docs  = $database->get_documents("mri-survey-feedback")->documents;
    $feedback = array();

    // Assign survey ID and partifipant into to feedback
    foreach ($feedback_docs as $entry) {

        $surveyId = $entry["surveyId"];

        $feedback[] = array(
            "surveyId"    => $surveyId,
            "feedback"    => $entry["feedback"],
            "participant" => isset($participants[$surveyId]) ? $participants[$surveyId] : null
        );
    }

    // Return response to user
    echo json_encode(array(
        "participants" => array_values($participants),
        "analytics" => $analytics,
        "feedback"  => $feedback
    ), JSON_PRETTY_PRINT);

?>