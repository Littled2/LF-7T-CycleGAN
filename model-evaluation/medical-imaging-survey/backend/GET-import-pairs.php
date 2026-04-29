<?php

    require_once __DIR__ . "/../../database/demand-db.php";
    require_once __DIR__ . "/../../database/demand-files.php";
    require_once __DIR__ . "/../../helpers/helpers.php";


    // =====================================
    //
    // For this script to work, there must be a configuration file called mri-survey-pairs.json in /resources/mri-survey
    //
    // =====================================
    
    
    must_be_authenticated();

    safely_start_session();

    http_method_must_be("GET");


    $pairs = json_decode(file_get_contents(__DIR__ . "/../../../resources/mri-survey/mri-survey-pairs.json"), TRUE);

    echo count($pairs) . " pairs found <br> ";

    $database = new DemandDB();

    for ($i=0; $i < count($pairs); $i++) { 
    
        $database->create_document("mri-survey-pairs", $pairs[$i]);

    }

    echo "Added " . count($pairs) . " pairs to database";

?>