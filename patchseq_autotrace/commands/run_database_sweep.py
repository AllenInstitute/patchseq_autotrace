import sqlite3
import time
import os
import argschema as ags

# This script is meant to sweep through an autotrace output directory and post-hoc update a sqlite .db file
class IO_Schema(ags.ArgSchema):
    database_file = ags.fields.OutputFile(description='.db file')
    autotrace_root_directory = ags.fields.InputDir(default="/allen/programs/celltypes/workgroups/mousecelltypes"
                                                           "/AutotraceReconstruction")


def main(args, **kwargs):
    database_file = args['database_file']
    autotrace_root_directory = args['autotrace_root_directory']

    con = sqlite3.connect(database_file)
    cur = con.cursor()

    # check that the specimen records table exists
    res = cur.execute("SELECT name FROM sqlite_master")
    if ("basic_post_hoc_specimen_records",) not in res.fetchall():
        cur.execute(
            "CREATE TABLE basic_post_hoc_specimen_records(specimen_id, has_raw_autotrace, autotrace_version, file_generated_timestamp)")

    res = cur.execute("SELECT * FROM basic_post_hoc_specimen_records")
    existing_specimen_records = res.fetchall()
    package_list = []

    cur.close()
    con.close()

    for spid in os.listdir(autotrace_root_directory):
        try:
            int(spid)
        except:
            # This is not a valid specimen ID if it is not an integer
            continue


        specimen_dir = os.path.join(autotrace_root_directory, spid)
        raw_swc_dir = os.path.join(specimen_dir, "SWC", "Raw")

        swc_files = []
        if os.path.exists(raw_swc_dir):
            swc_files = [f for f in os.listdir(raw_swc_dir) if f.endswith(".swc")]

        has_raw_swc = 0
        if swc_files:
            has_raw_swc = 1
            for fn in swc_files:
                # versioning of the patchseq_autotrace codebase
                # was incorporated into the file name so what was
                # previously saved as:
                # 1078838829_Aspiny1.0_1.0.swc
                # is now saved as:
                # 1078838829_Aspiny1.0_0.1.2_1.0.swc
                # where the _0.1.2_ represents the version of
                # patchseq_autotrace used to generate the file
                swc_pth = os.path.join(raw_swc_dir, fn)

                created_time = os.path.getctime(swc_pth)
                created_time = time.ctime(created_time)

                file_name_pieces = fn.split("_")
                if len(file_name_pieces) != 4:
                    version = "0"
                else:
                    version = file_name_pieces[2]

                package = (int(spid), has_raw_swc, version, created_time)

                if package not in existing_specimen_records:
                    package_list.append(package)

        else:
            version = "None"
            created_time = "None"
            package = (int(spid), has_raw_swc, version, created_time)
            if package not in existing_specimen_records:
                package_list.append(package)

    if package_list:
        con = sqlite3.connect(database_file)
        cur = con.cursor()

        insert_string = " \n".join([f"({p[0]}, {p[1]}, '{p[2]}', '{p[3]}')," for p in package_list])[:-1]  # last comma
        insert_cmd = f"""
            INSERT INTO basic_post_hoc_specimen_records VALUES
            {insert_string}
        """
        cur.execute(insert_cmd)
        con.commit()

        cur.close()
        con.close()


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
