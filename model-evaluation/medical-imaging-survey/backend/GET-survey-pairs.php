<?php

    require_once __DIR__ . "/../../database/demand-db.php";
    require_once __DIR__ . "/../../database/demand-files.php";
    require_once __DIR__ . "/../../helpers/helpers.php";

    http_method_must_be("GET");

    $database = new DemandDB();

    $pairs = $database
        ->get_documents("mri-survey-pairs")
        ->sort_by("created_at", "asc")
        ->documents;

    echo json_encode($pairs, JSON_PRETTY_PRINT);

?>