// Power Query (M): load the CSV star schema written by `python -m oilfield.bi.export`.
// Create a text parameter NptFolder pointing at the data/powerbi folder, then one query per
// table using LoadTable("fact_npt"), LoadTable("dim_cause"), ...
// On Databricks, connect to the SQL warehouse instead (Get Data > Azure Databricks) and read the
// same tables from the gold schema; the model and measures are unchanged.

let
    LoadTable = (name as text) as table =>
        let
            Source = Csv.Document(
                File.Contents(NptFolder & "\" & name & ".csv"),
                [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
            ),
            Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
            Typed =
                if name = "fact_npt" then
                    Table.TransformColumnTypes(Promoted, {
                        {"wellbore_key", Int64.Type}, {"date_key", Int64.Type}, {"cause_key", Int64.Type},
                        {"hours", type number}, {"confidence", type number}, {"md_m", type number}})
                else if name = "fact_activity" then
                    Table.TransformColumnTypes(Promoted, {
                        {"wellbore_key", Int64.Type}, {"date_key", Int64.Type}, {"hours", type number},
                        {"is_npt", Int64.Type}})
                else if name = "dim_date" then
                    Table.TransformColumnTypes(Promoted, {
                        {"date_key", Int64.Type}, {"date", type date}, {"year", Int64.Type},
                        {"month", Int64.Type}, {"iso_week", Int64.Type}})
                else if name = "dim_cause" then
                    Table.TransformColumnTypes(Promoted, {{"cause_key", Int64.Type}, {"sort_order", Int64.Type}})
                else if name = "dim_wellbore" then
                    Table.TransformColumnTypes(Promoted, {
                        {"wellbore_key", Int64.Type}, {"first_report", type date}, {"last_report", type date}})
                else Promoted
        in
            Typed
in
    LoadTable
