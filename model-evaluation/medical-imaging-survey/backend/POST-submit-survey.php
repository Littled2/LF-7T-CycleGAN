<?php

    require_once __DIR__ . "/../../database/demand-db.php";
    require_once __DIR__ . "/../../database/demand-files.php";
    require_once __DIR__ . "/../../helpers/helpers.php";

    http_method_must_be("POST");

    validate_request_data($_POST,  "surveyId", "low_field|string", "artefacts|string", "structure|string", "snr|string", "overall|string");

    $database = new DemandDB();

    // Write this response to the database

    $newDoc = array(
        "surveyId" => $_POST["surveyId"],
        "low_field" => $_POST["low_field"],
        "artefacts" => $_POST["artefacts"],
        "structure" => $_POST["structure"],
        "snr" => $_POST["snr"],
        "overall" => $_POST["overall"]
    );

    $newDocID = $database->create_document("mri-survey-responses", $newDoc);
    
    echo $newDocID;

?>