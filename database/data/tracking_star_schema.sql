-- =============================================================
-- Star Schema: Object Tracking Database
-- =============================================================

-- -------------------------------------------------------------
-- Dimension: dim_frame
-- One row per video frame captured
-- -------------------------------------------------------------
CREATE TABLE dim_frame (
    frame_id        SERIAL          PRIMARY KEY,
    frame_number    INTEGER         NOT NULL,
    source_video_id VARCHAR(255),
    fps             NUMERIC(6, 3),
    -- add path to source video for traceability
    video_path      VARCHAR(500),
    UNIQUE (frame_number, source_video_id)
);

-- -------------------------------------------------------------
-- Dimension: dim_track
-- One row per tracked object lifetime
-- -------------------------------------------------------------
CREATE TABLE dim_track (
    track_id          SERIAL    PRIMARY KEY,
    first_seen_frame  INTEGER,
    last_seen_frame   INTEGER
);

-- -------------------------------------------------------------
-- Dimension: dim_class
-- One row per object class label
-- -------------------------------------------------------------
CREATE TABLE dim_class (
    class_id        SERIAL        PRIMARY KEY,
    class_name      VARCHAR(100)  NOT NULL UNIQUE,
    class_category  VARCHAR(100)            -- e.g. 'vehicle', 'person'
);

-- -------------------------------------------------------------
-- Dimension: dim_time
-- One row per unique timestamp
-- -------------------------------------------------------------
CREATE TABLE dim_time (
    time_id       SERIAL          PRIMARY KEY,
    timestamp_s   NUMERIC(14, 6)  NOT NULL UNIQUE,
    hour          SMALLINT        NOT NULL,
    minute        SMALLINT        NOT NULL,
    second        SMALLINT        NOT NULL,
    millisecond   SMALLINT        NOT NULL
);

-- -------------------------------------------------------------
-- Fact: fact_detection
-- One row per detection event
-- -------------------------------------------------------------
CREATE TABLE fact_detection (
    detection_id        BIGSERIAL       PRIMARY KEY,
    frame_id            INTEGER         NOT NULL REFERENCES dim_frame(frame_id),
    track_id            INTEGER         NOT NULL REFERENCES dim_track(track_id),
    class_id            INTEGER         NOT NULL REFERENCES dim_class(class_id),
    time_id             INTEGER         NOT NULL REFERENCES dim_time(time_id),

    -- Bounding box (top-left / bottom-right corners)
    x1                  NUMERIC(10, 4)  NOT NULL,
    y1                  NUMERIC(10, 4)  NOT NULL,
    x2                  NUMERIC(10, 4)  NOT NULL,
    y2                  NUMERIC(10, 4)  NOT NULL,

    -- Derived bounding box measures
    width               NUMERIC(10, 4)  NOT NULL,
    height              NUMERIC(10, 4)  NOT NULL,

    -- Centroid
    cx                  NUMERIC(10, 4)  NOT NULL,
    cy                  NUMERIC(10, 4)  NOT NULL,

    -- Detection confidence state (e.g. 'tentative', 'confirmed', 'deleted')
    confidence_state    VARCHAR(50)     NOT NULL DEFAULT 'tentative'
);

-- =============================================================
-- Indexes
-- =============================================================

-- Fact table FK lookups
CREATE INDEX idx_fact_frame   ON fact_detection (frame_id);
CREATE INDEX idx_fact_track   ON fact_detection (track_id);
CREATE INDEX idx_fact_class   ON fact_detection (class_id);
CREATE INDEX idx_fact_time    ON fact_detection (time_id);

-- Common analytical queries
CREATE INDEX idx_fact_confidence  ON fact_detection (confidence_state);
CREATE INDEX idx_fact_cx_cy       ON fact_detection (cx, cy);

-- Dimension lookups
CREATE INDEX idx_class_name   ON dim_class (class_name);
CREATE INDEX idx_time_ts      ON dim_time  (timestamp_s);
CREATE INDEX idx_frame_num    ON dim_frame (frame_number);

-- =============================================================
-- Seed: dim_class with known labels from sample data
-- =============================================================
INSERT INTO dim_class (class_name, class_category) VALUES
    ('person', 'person'),
    ('car',    'vehicle'),
    ('truck',  'vehicle');