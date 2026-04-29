<?php

    require_once __DIR__ . "/../../database/demand-db.php";
    require_once __DIR__ . "/../../database/demand-files.php";
    require_once __DIR__ . "/../../helpers/helpers.php";

    http_method_must_be("POST");

    validate_request_data($_POST,  "surveyId", "feedback|string");

    $database = new DemandDB();

    // Write this response to the database

    $newDoc = array(
        "surveyId" => $_POST["surveyId"],
        "feedback" => $_POST["feedback"]
    );

    $newDocID = $database->create_document("mri-survey-feedback", $newDoc);
    
    echo $newDocID;

?>