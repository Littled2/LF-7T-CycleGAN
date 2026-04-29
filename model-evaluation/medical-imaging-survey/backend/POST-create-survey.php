<?php

    require_once __DIR__ . "/../../database/demand-db.php";
    require_once __DIR__ . "/../../database/demand-files.php";
    require_once __DIR__ . "/../../helpers/helpers.php";

    http_method_must_be("POST");

    validate_request_data($_POST,  "name|string", "jobTitle|string", "email|string");

    $database = new DemandDB();

    $newDoc = array(
        "email" => $_POST["email"],
        "name" => $_POST["name"],
        "jobTitle" => $_POST["jobTitle"]
    );

    $newDocID = $database->create_document("mri-surveys", $newDoc);
    
    echo $newDocID;

?>