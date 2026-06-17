-- =============================================================
-- Star Schema: Object Tracking Database (v3)
-- Tracklet-centered with dim_track, dim_frame, dim_camera,
-- dim_video, dim_class, dim_time
-- =============================================================

-- -------------------------------------------------------------
-- Dimension: dim_camera
-- One row per physical camera
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_camera (
    camera_id     SERIAL        PRIMARY KEY,
    camera_name   VARCHAR(255)  NOT NULL UNIQUE,
    location      VARCHAR(255),
    fps           NUMERIC(6, 3),
    resolution    VARCHAR(50)
);

-- -------------------------------------------------------------
-- Dimension: dim_video
-- One row per video segment per camera
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_video (
    video_id         SERIAL          PRIMARY KEY,
    camera_id        INTEGER         NOT NULL REFERENCES dim_camera(camera_id),
    segment_number   INTEGER         NOT NULL,
    start_ts         NUMERIC(14, 6),
    end_ts           NUMERIC(14, 6),
    file_path        VARCHAR(500),
    UNIQUE (camera_id, segment_number)
);

-- -------------------------------------------------------------
-- Dimension: dim_class
-- One row per object class label
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_class (
    class_id        SERIAL        PRIMARY KEY,
    class_name      VARCHAR(100)  NOT NULL UNIQUE,
    class_category  VARCHAR(100),
    model_version   VARCHAR(50)
);

-- -------------------------------------------------------------
-- Dimension: dim_time
-- One row per unique timestamp
-- Shared by start_time_id and end_time_id on fact_tracklet
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_time (
    time_id       SERIAL          PRIMARY KEY,
    timestamp_s   NUMERIC(14, 6)  NOT NULL UNIQUE,
    hour          SMALLINT        NOT NULL,
    minute        SMALLINT        NOT NULL,
    second        SMALLINT        NOT NULL,
    millisecond   SMALLINT        NOT NULL
);

-- -------------------------------------------------------------
-- Dimension: dim_track
-- One row per real-world object (groups multiple tracklets)
-- avg_speed_px_per_s: detection-count-weighted mean across all
--   child tracklets; NULL until at least one tracklet has speed
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_track (
    track_id              SERIAL          PRIMARY KEY,
    camera_id             INTEGER         NOT NULL REFERENCES dim_camera(camera_id),
    first_seen_ts         NUMERIC(14, 6),
    last_seen_ts          NUMERIC(14, 6),
    tracklet_count        INTEGER         NOT NULL DEFAULT 0,
    avg_speed_px_per_s    NUMERIC(10, 4)           -- weighted mean over child tracklets; NULL until computed
);

-- -------------------------------------------------------------
-- Dimension: dim_frame
-- One row per video frame
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_frame (
    frame_id        SERIAL          PRIMARY KEY,
    video_id        INTEGER         NOT NULL REFERENCES dim_video(video_id),
    frame_number    INTEGER         NOT NULL,
    timestamp_s     NUMERIC(14, 6)  NOT NULL,
    UNIQUE (video_id, frame_number)
);

-- -------------------------------------------------------------
-- Fact: fact_tracklet
-- One row per continuous tracked object segment
-- Class assigned at tracklet level
-- avg_speed_px_per_s: mean of child detection instant speeds;
--   NULL when detection_count < 2
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_tracklet (
    tracklet_id           BIGSERIAL       PRIMARY KEY,
    track_id              INTEGER         NOT NULL REFERENCES dim_track(track_id),
    camera_id             INTEGER         NOT NULL REFERENCES dim_camera(camera_id),
    video_id              INTEGER         NOT NULL REFERENCES dim_video(video_id),
    class_id              INTEGER         NOT NULL REFERENCES dim_class(class_id),
    start_time_id         INTEGER         NOT NULL REFERENCES dim_time(time_id),
    end_time_id           INTEGER         NOT NULL REFERENCES dim_time(time_id),

    source_track_id       INTEGER         NOT NULL,   -- original tracker ID (not globally unique)
    confidence_state      VARCHAR(50)     NOT NULL DEFAULT 'tentative',
    detection_count       INTEGER         NOT NULL DEFAULT 0,
    avg_speed_px_per_s    NUMERIC(10, 4),             -- mean of child detection instant speeds; NULL when detection_count < 2
    embedding             FLOAT[]
);

-- -------------------------------------------------------------
-- Child fact: fact_detection
-- One row per bounding box observation within a tracklet
-- Class inherited from fact_tracklet via join
-- speed_px_per_s: Euclidean centroid displacement from the
--   previous detection divided by elapsed time; NULL for the
--   first detection of each tracklet
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_detection (
    detection_id      BIGSERIAL       PRIMARY KEY,
    tracklet_id       BIGINT          NOT NULL REFERENCES fact_tracklet(tracklet_id),
    frame_id          INTEGER         NOT NULL REFERENCES dim_frame(frame_id),
    timestamp_s       NUMERIC(14, 6)  NOT NULL,

    x1                NUMERIC(10, 4)  NOT NULL,
    y1                NUMERIC(10, 4)  NOT NULL,
    x2                NUMERIC(10, 4)  NOT NULL,
    y2                NUMERIC(10, 4)  NOT NULL,
    width             NUMERIC(10, 4)  NOT NULL,
    height            NUMERIC(10, 4)  NOT NULL,
    cx                NUMERIC(10, 4)  NOT NULL,
    cy                NUMERIC(10, 4)  NOT NULL,
    speed_px_per_s    NUMERIC(10, 4)             -- instantaneous speed: sqrt(Δcx²+Δcy²)/Δt vs previous detection; NULL for first detection in tracklet
);

-- =============================================================
-- Indexes
-- =============================================================

-- dim_video
CREATE INDEX IF NOT EXISTS idx_video_camera          ON dim_video    (camera_id);

-- dim_frame
CREATE INDEX IF NOT EXISTS idx_frame_video           ON dim_frame    (video_id);
CREATE INDEX IF NOT EXISTS idx_frame_ts              ON dim_frame    (timestamp_s);

-- dim_track
CREATE INDEX IF NOT EXISTS idx_track_camera          ON dim_track    (camera_id);

-- dim_time
CREATE INDEX IF NOT EXISTS idx_time_ts               ON dim_time     (timestamp_s);

-- fact_tracklet
CREATE INDEX IF NOT EXISTS idx_tracklet_track        ON fact_tracklet (track_id);
CREATE INDEX IF NOT EXISTS idx_tracklet_camera       ON fact_tracklet (camera_id);
CREATE INDEX IF NOT EXISTS idx_tracklet_video        ON fact_tracklet (video_id);
CREATE INDEX IF NOT EXISTS idx_tracklet_class        ON fact_tracklet (class_id);
CREATE INDEX IF NOT EXISTS idx_tracklet_start_time   ON fact_tracklet (start_time_id);
CREATE INDEX IF NOT EXISTS idx_tracklet_end_time     ON fact_tracklet (end_time_id);
CREATE INDEX IF NOT EXISTS idx_tracklet_confidence   ON fact_tracklet (confidence_state);
CREATE INDEX IF NOT EXISTS idx_tracklet_source_track ON fact_tracklet (source_track_id);

-- fact_detection
CREATE INDEX IF NOT EXISTS idx_detection_tracklet    ON fact_detection (tracklet_id);
CREATE INDEX IF NOT EXISTS idx_detection_frame       ON fact_detection (frame_id);
CREATE INDEX IF NOT EXISTS idx_detection_ts          ON fact_detection (timestamp_s);
CREATE INDEX IF NOT EXISTS idx_detection_centroid    ON fact_detection (cx, cy);

-- =============================================================
-- Seed: dim_class with known labels
-- =============================================================
INSERT INTO dim_class (class_name, class_category) VALUES
    ('person',  'person'),
    ('car',     'vehicle'),
    ('truck',   'vehicle'),
    ('bus',     'vehicle'),
    ('bicycle', 'vehicle'),
    ('motorbike', 'vehicle')
ON CONFLICT (class_name) DO NOTHING;
